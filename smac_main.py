# multiagent_main.py
import importlib.util
import os, json, time, random
import concurrent.futures

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp, numpy as np, tqdm, wandb
from absl import app, flags
from ml_collections import config_flags, config_dict

from envs.environments import get_environment
from utils.replay_buffers import FlashbaxReplayBuffer
from vault_utils.download_vault import download_and_unzip_vault
from utils.loggers import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    setup_wandb,
    get_wandb_video,
)

FLAGS = flags.FLAGS
# ------------------------------------- 기본 플래그 -------------------------------------------------
flags.DEFINE_string ('run_group',       'Debug', 'Run group')
flags.DEFINE_integer('seed',            0,       'Random seed')
flags.DEFINE_string ('env',             'smac_v1', 'env')
flags.DEFINE_string ('source',          'og_marl', 'builder')
flags.DEFINE_string ('scenario',        '3m', 'scenario')
flags.DEFINE_string ('dataset',         'Good',  'quality')
flags.DEFINE_string ('save_dir',        'exp/',    'Save directory')
flags.DEFINE_string('agent_name',       'scout',      'Agent name')
flags.DEFINE_string('project_name',       'test',   'WandB project name')
flags.DEFINE_string('data_dir',       './data/',    'Directory to store datasets')

flags.DEFINE_integer('offline_steps',    1_000_000, 'Offline gradient steps')
flags.DEFINE_integer('sequence_length',  20,        'Replay sequence length')
flags.DEFINE_integer('sample_period',  1,        'Sample period')
flags.DEFINE_integer('buffer_size',      2_000_000, 'Max transitions in buffer')
flags.DEFINE_integer('batch_size',      32, 'Batch size')

flags.DEFINE_integer('log_interval',  50_000,   '')
flags.DEFINE_integer('eval_interval', 100_000,  '')
flags.DEFINE_integer('save_interval', 1_000_001, '')
flags.DEFINE_integer('eval_video_episodes', 1, 'Number of evaluation episodes to record as video.')
flags.DEFINE_integer('eval_video_frame_skip', 1, 'Render every k-th frame when recording evaluation videos.')
flags.DEFINE_integer('num_eval_workers', 10, 'Number of parallel workers for evaluation (threads).')
# -----------------------------------------------------------------------------------------------

config_flags.DEFINE_config_file('agent', 'agents/discrete_macflow.py', lock_config=False)
flags.DEFINE_alias('agent_config', 'agent')

def main(_):
    agent_cfg = FLAGS.agent
    if hasattr(agent_cfg, 'copy_and_resolve_references'):
        cfg = agent_cfg.copy_and_resolve_references()
    else:
        cfg = config_dict.ConfigDict(agent_cfg.to_dict())
    requested_agent_name = FLAGS.agent_name
    agent_name = 'scout' if requested_agent_name == 'discrete_macflow_vgf' else requested_agent_name
    cfg['agent_name'] = agent_name

    if agent_name == 'macflow':
        from agents.discrete_macflow import MACFlowDiscreteAgent
        agent_cls = MACFlowDiscreteAgent
    elif agent_name == 'scout':
        from agents.discrete_scout import ScoutAgent
        agent_cls = ScoutAgent
    else:
        raise ValueError(f"Unknown agent_name '{requested_agent_name}'. Please add it to AGENT_CONFIG_MAP and import here.")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=exp_name)

    save_root = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(save_root, exist_ok=True)
    json.dump(get_flag_dict(), open(os.path.join(save_root, 'flags.json'), 'w'))

    # -------- ENV & Buffer ----------------------------------- --------------------------
    env_factory = lambda s: get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, s)
    env = env_factory(FLAGS.seed)
    agent_names = list(env.agents)

    buffer = FlashbaxReplayBuffer(
        sequence_length=FLAGS.sequence_length,
        batch_size=FLAGS.batch_size,
        sample_period=FLAGS.sample_period,
        seed=FLAGS.seed,
        max_size=FLAGS.buffer_size)

    if FLAGS.data_dir == '/datastor1/dongsu':
        download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario,
                                 dataset_base_dir=FLAGS.data_dir)
        buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset),
                                   rel_dir=FLAGS.data_dir)
    else:
        download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario)
        buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset))

    example = buffer.sample()
    ex_obs  = jnp.asarray(example['observations'])
    ex_act = jnp.asarray(example['actions'])
    if len(ex_act.shape) == 3:
        ex_act = jnp.asarray(example["infos"]["legals"])
    ex_state = jnp.asarray(example['infos']['state'])

    agent = agent_cls.create(
        seed=FLAGS.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        agent_names=agent_names,
        config=cfg,
    )

    # -------- Logger -----------------------------------------------------------------------
    train_csv = CsvLogger(os.path.join(save_root, 'train.csv'))
    eval_csv  = CsvLogger(os.path.join(save_root, 'eval.csv'))
    t0 = time.time(); last = t0

    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), dynamic_ncols=True):
        if step <= FLAGS.offline_steps:
            batch = buffer.sample()

        agent, info = agent.update(batch, step)

        if step % FLAGS.log_interval == 0:
            metrics = {f'train/{k}': float(v) for k, v in info.items()}
            metrics['time/iter_s'] = (time.time() - last) / FLAGS.log_interval
            wandb.log(metrics, step=step)
            train_csv.log(metrics, step=step)
            last = time.time()

        if step == 1 or step % FLAGS.eval_interval == 0:
            eval_ret = _evaluate(agent, env, n_eps=10, seed=FLAGS.seed)
            wandb.log(eval_ret, step=step);  eval_csv.log(eval_ret, step=step)

        if step % FLAGS.save_interval == 0:
            agent.network.save(os.path.join(save_root, f'ckpt_{step}.npz'))

    train_csv.close(); eval_csv.close()

# === Helper ======================================================================================
def _evaluate(agent, env, n_eps=10, seed=0):
    episode_returns = []
    for _ in range(n_eps):
        observations, infos = env.reset()

        done = False
        episode_return = 0.0
        carry = None
        reset_mask = None
        while not done:
            # t0 = time.time()
            # Use stateful LSTM carry when enabled; reset at episode start or per-agent terminal
            actions, carry = agent.sample_actions_with_carry(
                observations,
                carry,
                jax.random.PRNGKey(seed),
                infos.get('legals'),
                reset_mask=reset_mask,
            )

            actions = {
                agent_name: int(actions[agent_name].item())
                for agent_name in agent.agent_names
            }

            observations, rewards, terminal, truncation, infos = env.step(actions)
            # t1 = time.time()
            # print(t1 - t0)
            episode_return += np.mean(list(rewards.values()), dtype="float")

            # Build per-agent reset mask for next step's RNN carry, if any agent terminated
            reset_mask = jnp.asarray([terminal[ag] for ag in agent.agent_names], dtype=jnp.bool_)
            done = all(terminal.values()) or all(truncation.values())

        episode_returns.append(episode_return)

    logs = {
        "evaluation/mean_episode_return": np.mean(episode_returns),
        "evaluation/max_episode_return": np.max(episode_returns),
        "evaluation/min_episode_return": np.min(episode_returns),
    }
    return logs

# ===============================================================================================

if __name__ == '__main__':
    app.run(main)

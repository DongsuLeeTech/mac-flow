# multiagent_main.py
import os, json, time, random

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp, numpy as np, tqdm, wandb
from absl import app, flags
from ml_collections import config_flags

from envs.environments import get_environment
from utils.replay_buffers import FlashbaxReplayBuffer
from vault_utils.download_vault import download_and_unzip_vault
from utils.loggers import (CsvLogger, get_exp_name,
                             get_flag_dict, setup_wandb)

FLAGS = flags.FLAGS
# ------------------------------------- 기본 플래그 -------------------------------------------------
flags.DEFINE_string ('run_group',       'Debug', 'Run group')
flags.DEFINE_integer('seed',            0,       'Random seed')
flags.DEFINE_string ('env',             'mamujoco', 'env')
flags.DEFINE_string ('source',          'omiga', 'builder')
flags.DEFINE_string ('scenario',        '3hopper', 'scenario')
flags.DEFINE_string ('dataset',         'Expert',  'quality')
flags.DEFINE_string ('save_dir',        'exp/',    'Save directory')
flags.DEFINE_string('agent_name',       'macflow_vgf',      'Agent name')
flags.DEFINE_string('project_name',       'test',   'WandB project name')
flags.DEFINE_string('data_dir',       '/datastor1/dongsu',    'Directory to store datasets')

flags.DEFINE_integer('offline_steps',    500_000, 'Offline gradient steps')
flags.DEFINE_integer('sequence_length',  20,        'Replay sequence length')
flags.DEFINE_integer('sample_period',  1,        'Sample period')
flags.DEFINE_integer('buffer_size',      2_000_000, 'Max transitions in buffer')
flags.DEFINE_integer('batch_size',      32, 'Batch size')

flags.DEFINE_integer('log_interval',  5_000,   '')
flags.DEFINE_integer('eval_interval',  50_000,  '')
flags.DEFINE_integer('save_interval', 500_001, '')
# -----------------------------------------------------------------------------------------------

def main(_):
    from agents.macflow import MACFlowAgent
    config_flags.DEFINE_config_file('agent', 'agents/macflow.py', lock_config=False)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=exp_name)

    save_root = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(save_root, exist_ok=True)
    json.dump(get_flag_dict(), open(os.path.join(save_root, 'flags.json'), 'w'))

    # -------- ENV & Buffer ----------------------------------- --------------------------
    env = get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, FLAGS.seed)
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
    ex_state = jnp.asarray(example['infos']['state'])

    cfg = FLAGS.agent
    agent = MACFlowAgent.create(
        seed=FLAGS.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        agent_names=agent_names,
        config=cfg,)

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
        while not done:
            # For continuous control, pass actions as float arrays.
            actions = agent.sample_actions(observations, jax.random.PRNGKey(seed))

            # if FLAGS.scenario == 'simple_tag' or 'simple_world':
            #     random_actions = np.random.uniform(-1, 1, 2)
            #     actions['agent_3']

            observations, rewards, terminal, truncation, infos = env.step(actions)
            episode_return += np.mean(list(rewards.values()), dtype="float")

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
    os.environ["SUPPRESS_GR_PROMPT"] = "1"
    app.run(main)

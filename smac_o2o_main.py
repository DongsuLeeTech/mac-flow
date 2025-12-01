# multiagent_main.py
import os, json, time, random

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
flags.DEFINE_string ('env',             'smac_v2', 'env')
flags.DEFINE_string ('source',          'og_marl', 'builder')
flags.DEFINE_string ('scenario',        'terran_5_vs_5', 'scenario')
flags.DEFINE_string ('dataset',         'Replay',  'quality')
flags.DEFINE_string ('save_dir',        'exp/',    'Save directory')
flags.DEFINE_string('agent_name',       'diffbc',      'Agent name')

flags.DEFINE_integer('offline_steps', 500_000, 'Offline gradient steps')
flags.DEFINE_integer('online_steps',    500_000, 'Offline gradient steps')
flags.DEFINE_integer('sequence_length',  20,        'Replay sequence length')
flags.DEFINE_integer('sample_period',  1,        'Sample period')
flags.DEFINE_integer('buffer_size',      2_000_000, 'Max transitions in buffer')
flags.DEFINE_integer('batch_size',      32, 'Batch size')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

flags.DEFINE_integer('log_interval',  5_000,   '')
flags.DEFINE_integer('eval_interval', 100_000,  '')
flags.DEFINE_integer('save_interval', 1_000_000, '')
# -----------------------------------------------------------------------------------------------

def main(_):
    if FLAGS.agent_name == 'bc':
        from agents.bc import MABCAgent
        config_flags.DEFINE_config_file('agent', 'agents/bc.py', lock_config=False)
    elif FLAGS.agent_name == 'omar':
        from FPF.agents.discrete_icq import OMARAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_icq.py', lock_config=False)
    elif FLAGS.agent_name == 'discrete_omar':
        from agents.discrete_omar import DisOMARAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_omar.py', lock_config=False)
    elif FLAGS.agent_name == 'qmix':
        from agents.outdated.qmix import QMixAgent
        config_flags.DEFINE_config_file('agent', 'agents/qmix.py', lock_config=False)
    elif FLAGS.agent_name == 'discrete_bc':
        from agents.discrete_bc import MADisBCAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_bc.py', lock_config=False)
    elif FLAGS.agent_name == 'omiga':
        from FPF.agents.omiga import OMIGAAgent
        config_flags.DEFINE_config_file('agent', 'agents/omiga.py', lock_config=False)
    elif FLAGS.agent_name == 'bcq':
        from agents.discrete_bcq import MABCQAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_bcq.py', lock_config=False)
    elif FLAGS.agent_name == 'cql':
        from agents.discrete_cql import MACQLAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_cql.py', lock_config=False)
    elif FLAGS.agent_name == 'madiff':
        from agents.madiff import MADiffAgent
        config_flags.DEFINE_config_file('agent', 'agents/madiff.py', lock_config=False)
    if FLAGS.agent_name == 'diffbc':
        from agents.discrete_diffusion_bc import DiscreteDiffusionBCAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_diffusion_bc.py', lock_config=False)
    elif FLAGS.agent_name == 'flowbc':
        from agents.discrete_flowbc import FlowBCDiscreteAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_flowbc.py', lock_config=False)
    elif FLAGS.agent_name == 'madflow':
        from agents.discrete_macflow import MADFlowDiscreteAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_macflow.py', lock_config=False)
    elif FLAGS.agent_name == 'madflow_v3':
        from agents.discrete_madflow_v3 import MADFlowDiscreteAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_madflow_v3.py', lock_config=False)
    elif FLAGS.agent_name == 'fmc':
        from agents.FM import FMAgent
        config_flags.DEFINE_config_file('agent', 'agents/FM.py', lock_config=False)
    elif FLAGS.agent_name == 'fql':
        from agents.fql import FQLAgent
        config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)
    elif FLAGS.agent_name == 'icq':
        from agents.discrete_icq import MAICQAgent
        config_flags.DEFINE_config_file('agent', 'agents/discrete_icq.py', lock_config=False)
    elif FLAGS.agent_name == 'fql':
        from agents.fql import FQLAgent
        config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='FPF', group=FLAGS.run_group, name=exp_name)

    save_root = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(save_root, exist_ok=True)
    json.dump(get_flag_dict(), open(os.path.join(save_root, 'flags.json'), 'w'))

    # -------- ENV & Buffer ----------------------------------- --------------------------
    env = get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, FLAGS.seed)
    eval_env = get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, FLAGS.seed)
    agent_names = list(env.agents)

    buffer = FlashbaxReplayBuffer(
        sequence_length=FLAGS.sequence_length,
        batch_size=FLAGS.batch_size,
        sample_period=FLAGS.sample_period,
        seed=FLAGS.seed,
        max_size=FLAGS.buffer_size)

    online_buffer = FlashbaxReplayBuffer(
        sequence_length=FLAGS.sequence_length,
        batch_size=FLAGS.batch_size,
        sample_period=FLAGS.sample_period,
        seed=FLAGS.seed,
        max_size=FLAGS.buffer_size)

    download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario)
    buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset))

    example = buffer.sample()
    ex_obs  = jnp.asarray(example['observations'])
    ex_act = jnp.asarray(example['actions'])
    if len(ex_act.shape) == 3:
        ex_act = jnp.asarray(example["infos"]["legals"])
    ex_state = jnp.asarray(example['infos']['state'])

    cfg = FLAGS.agent
    if FLAGS.agent_name == 'bc':
        agent = MABCAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'omar':
        agent = OMARAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_states=ex_state,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'discrete_bc':
        agent = MADisBCAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'discrete_omar':
        agent = DisOMARAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_states=ex_state,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'qmix':
        agent = QMixAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_state=ex_state,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'omiga':
        agent = OMIGAAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_state=ex_state,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'madiff':
        agent = MADiffAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    if FLAGS.agent_name == 'diffbc':
        agent = DiscreteDiffusionBCAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'flowbc':
        agent = FlowBCDiscreteAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'madflow':
        agent = MADFlowDiscreteAgent.create(
            seed=FLAGS.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg, )
    elif FLAGS.agent_name == 'fmc':
        agent = FMAgent.create(
            seed=FLAGS.seed,
            ex_states=ex_state,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'fql':
        agent = FQLAgent.create(
            seed=FLAGS.seed,
            ex_states=ex_state,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'icq':
        agent = MAICQAgent.create(
            seed=FLAGS.seed,
            ex_states=ex_state,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'bcq':
        agent = MABCQAgent.create(
            seed=FLAGS.seed,
            ex_states=ex_state,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'cql':
        agent = MACQLAgent.create(
            seed=FLAGS.seed,
            ex_states=ex_state,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)
    elif FLAGS.agent_name == 'fql':
        agent = FQLAgent.create(
            seed=FLAGS.seed,
            ex_states=ex_state,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,)

    # -------- Logger -----------------------------------------------------------------------
    train_csv = CsvLogger(os.path.join(save_root, 'train.csv'))
    eval_csv  = CsvLogger(os.path.join(save_root, 'eval.csv'))
    t0 = time.time(); last = t0

    done = True
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), dynamic_ncols=True):
        if step <= FLAGS.offline_steps:
            batch = buffer.sample()
            agent, info = agent.update(batch, step)
        else:
            online_rng, key = jax.random.split(online_rng)

            if done:
                ob, infos = env.reset()
            if FLAGS.agent_name == 'madflow':
                actions = agent.sample_actions(ob, key, infos['legals'], temperature=1.0)
            else:
                actions = agent.sample_actions(ob, key, infos['legals'])
            actions = {agent_name: int(actions[agent_name].item()) for agent_name in agent.agent_names}

            next_ob, rewards, terminal, truncation, next_infos = env.step(actions)

            done = all(terminal.values()) or all(truncation.values())

            online_buffer.add(ob, actions, rewards, terminal, truncation, infos)

            ob = next_ob
            infos = next_infos

            if step - FLAGS.offline_steps > FLAGS.batch_size * FLAGS.sequence_length:
                if FLAGS.balanced_sampling:
                    dataset_batch = buffer.sample()
                    replay_batch = online_buffer.sample()
                    batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
                else:
                    batch = online_buffer.sample()

                agent, info = agent.update(batch, step)

        if step % FLAGS.log_interval == 0:
            metrics = {f'train/{k}': float(v) for k, v in info.items()}
            metrics['time/iter_s'] = (time.time() - last) / FLAGS.log_interval
            wandb.log(metrics, step=step)
            train_csv.log(metrics, step=step)
            last = time.time()

        if step == 1 or step % FLAGS.eval_interval == 0:
            eval_ret = _evaluate(agent, eval_env, n_eps=10, seed=FLAGS.seed)
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
            actions = agent.sample_actions(observations, jax.random.PRNGKey(seed), infos['legals'])

            actions = {
                agent_name: int(actions[agent_name].item())
                for agent_name in agent.agent_names
            }

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
    app.run(main)

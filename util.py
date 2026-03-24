from __future__ import annotations
import os
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np

def set_growing_gpu_memory() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def gather(values: jnp.ndarray,
           indices: jnp.ndarray,
           axis: int = -1,
           keepdims: bool = False) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(indices, values.shape[axis], dtype=values.dtype)
    if values.ndim > 4:
        one_hot = jnp.expand_dims(one_hot, axis=-1)
    gathered = jnp.sum(values * one_hot, axis=axis, keepdims=keepdims)
    return gathered


def switch_two_leading_dims(x: jnp.ndarray) -> jnp.ndarray:
    trailing = list(range(2, x.ndim))
    return jnp.transpose(x, axes=[1, 0, *trailing])

def merge_batch_and_agent_dim_of_time_major_sequence(x: jnp.ndarray) -> jnp.ndarray:
    T, B, N, *rest = x.shape
    return jnp.reshape(x, (T, B * N, *rest))

def merge_time_batch_and_agent_dim(x: jnp.ndarray) -> jnp.ndarray:
    T, B, N, *rest = x.shape
    return jnp.reshape(x, (T * B * N, *rest))

def expand_time_batch_and_agent_dim_of_time_major_sequence(x: jnp.ndarray, T: int, B: int, N: int) -> jnp.ndarray:
    assert x.shape[0] == T * B * N
    rest = x.shape[1:]
    return jnp.reshape(x, (T, B, N, *rest))

def expand_batch_and_agent_dim_of_time_major_sequence(x: jnp.ndarray, B: int, N: int) -> jnp.ndarray:
    T, BN, *rest = x.shape
    assert BN == B * N
    return jnp.reshape(x, (T, B, N, *rest))


def concat_agent_id_to_obs(obs: jnp.ndarray, agent_id: int, N: int) -> jnp.ndarray:
    is_vector = (obs.ndim == 1)
    if is_vector:
        agent_emb = jax.nn.one_hot(agent_id, N, dtype=obs.dtype)
    else:
        h, w = obs.shape[:2]
        agent_emb = jnp.zeros((h, w, 1), obs.dtype) + agent_id / N + 1 / (2 * N)
    if (not is_vector) and obs.ndim == 2:
        obs = obs[..., None]
    return jnp.concatenate([agent_emb, obs], axis=-1)

def unroll_rnn(rnn, inputs: jnp.ndarray, resets: jnp.ndarray):
    def _step(carry, inp):
        x_t, reset_t = inp
        h = carry
        y, h_next = rnn(x_t, h)
        h_next = jnp.where(reset_t[..., None], rnn.initial_state(x_t.shape[0]), h_next)
        return h_next, y
    init_h = rnn.initial_state(inputs.shape[1])
    _, ys = lax.scan(_step, init_h, (inputs, resets))
    return ys

def batch_concat_agent_id_to_obs(obs: jnp.ndarray) -> jnp.ndarray:
    B, T, N = obs.shape[:3]
    is_vector = (obs.ndim == 4)

    if is_vector:
        agent_ids = jax.nn.one_hot(jnp.arange(N), N, dtype=obs.dtype)
    else:
        h, w = obs.shape[3:5]
        agent_ids = jnp.zeros((N, h, w, 1), obs.dtype)
        agent_ids = agent_ids + (jnp.arange(N)[:, None, None, None] / N + 1 / (2 * N))

    agent_ids = jnp.broadcast_to(agent_ids, (B, T, N, *agent_ids.shape[1:]))

    if (not is_vector) and obs.ndim == 5:
        obs = obs[..., None]
    return jnp.concatenate([agent_ids, obs], axis=-1)


def batched_agents(agents: Sequence[str],
                   batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "truncations": [],
        "infos": {},
    }

    for ag in agents:
        for k in out:
            if k == "infos":
                continue
            out[k].append(batch_dict[k][ag])
    for k in out:
        if k == "infos":
            continue
        out[k] = jnp.stack(out[k], axis=2)

    out["terminals"]   = out["terminals"].astype(jnp.float32)
    out["truncations"] = out["truncations"].astype(jnp.float32)

    # optional infos
    infos_in = batch_dict.get("infos", {})
    if "legals" in infos_in:
        out["infos"]["legals"] = jnp.stack([infos_in["legals"][ag] for ag in agents], axis=2)
    if "state" in infos_in:
        out["infos"]["state"] = jnp.asarray(infos_in["state"], dtype=jnp.float32)

    # mask
    if "mask" in infos_in:
        out["mask"] = jnp.asarray(infos_in["mask"], dtype=jnp.float32)
    else:
        out["mask"] = jnp.ones_like(out["terminals"][:, :, 0], dtype=jnp.float32)
    return out
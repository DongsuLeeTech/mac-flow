"""
Run examples:
  Quadratic: python landmark_covering.py --toy quadratic --steps 2000 --save_dir exp/toy_quadratic
  Ring:      python landmark_covering.py --toy ring      --steps 2000 --save_dir exp/toy_ring
  XOR:       python landmark_covering.py --toy xor       --steps 2000 --coupling_c 1.0 --noise_p 0.0 --imbalance_q 0.5 --save_dir exp/toy_xor
  Landmark:  python landmark_covering.py --toy landmark  --steps 2000 --lm_T 50 --lm_noise_sigma 0.02 --lm_swap_q 0.0 --agent macflow --lm_reward_c 10.0 --save_dir exp/toy_landmark
"""

import argparse
import os
import sys
import time
from typing import Tuple, Dict, Any, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.macflow import MACFlowAgent, get_config as get_fact_cfg
try:
    from agents.diffusion_bc import DiffusionBCAgent, get_config as get_diff_cfg
except Exception:
    DiffusionBCAgent = None
    def get_diff_cfg():
        raise ImportError('agents/diffusion_bc.py not available')
from utils.loggers import CsvLogger, get_exp_name
from util import batch_concat_agent_id_to_obs, switch_two_leading_dims

# === Plot style constants ===
POLICY_COLOR = "#BF5700"   # Policy-related metrics
VALUE_COLOR = "#005F86"    # Value-related metrics
OTHER_COLOR = "#333F48"    # Other metrics / references


# ============ Env Q definitions (JAX) ============ #

def q_env_quadratic(a: jnp.ndarray) -> jnp.ndarray:
    a1, a2 = a[..., 0], a[..., 1]
    return - (a1 - a2) ** 2 - 0.1 * (a1 ** 2 + a2 ** 2)


def q_env_ring(a: jnp.ndarray) -> jnp.ndarray:
    r2 = jnp.sum(a ** 2, axis=-1)
    return - (r2 - 1.0) ** 2


def estimate_env_lipschitz(q_fn, grid_size: int = 81) -> float:
    """Estimate sup ||∇Q|| over grid on [-1,1]^2."""
    xs = jnp.linspace(-1.0, 1.0, grid_size)
    A1, A2 = jnp.meshgrid(xs, xs, indexing='ij')
    pts = jnp.stack([A1.reshape(-1), A2.reshape(-1)], axis=-1)  # (B,2)

    def grad_norm(a):
        g = jax.grad(lambda z: q_fn(z).squeeze())(a)
        return jnp.linalg.norm(g)

    norms = jax.vmap(grad_norm)(pts)
    return float(jnp.max(norms))


def estimate_critic_lipschitz(agent: MACFlowAgent, grid_size: int = 81) -> float:
    # Build dummy observation (B=1, T=1, N=2, O=1) then append agent IDs
    obs = jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)
    obs = batch_concat_agent_id_to_obs(obs)  # (B, T, N, O')

    @jax.jit
    def q_critic(a_vec):
        a1, a2 = a_vec[0], a_vec[1]
        # Actions (B=1, T=1, N=2, A=1) and switch to (T,B,...) to match training path
        actions_bt = jnp.array([[[[a1], [a2]]]], dtype=jnp.float32)
        obs_tb = switch_two_leading_dims(obs)
        actions_tb = switch_two_leading_dims(actions_bt)
        qs = agent.network.select('critic')(obs_tb, actions=actions_tb)  # (E, T=1, B=1, N=2)
        q_mixed = qs.mean(axis=0).mean(axis=-1)[0, 0]  # scalar
        return q_mixed

    @jax.jit
    def grad_norm(a_vec):
        g = jax.grad(q_critic)(a_vec)
        return jnp.linalg.norm(g)

    xs = jnp.linspace(-1.0, 1.0, grid_size)
    A1, A2 = jnp.meshgrid(xs, xs, indexing='ij')
    pts = jnp.stack([A1.reshape(-1), A2.reshape(-1)], axis=-1)
    norms = jax.vmap(grad_norm)(pts)
    return float(jnp.max(norms))


def estimate_critic_lipschitz_nd(agent: MACFlowAgent,
                                 obs_template: Optional[jnp.ndarray] = None,
                                 num_points: int = 128,
                                 seed: int = 0) -> float:
    """Estimate sup ||∇_a Q_critic|| over [-1,1]^{N*A} for the learned critic.

    Uses random sampling of action vectors in the hypercube, conditioned on a
    fixed observation template (B=1,T=1).
    """
    if obs_template is None:
        N = int(agent.config['num_agents'])
        obs = jnp.zeros((1, 1, N, 1), dtype=jnp.float32)
    else:
        obs = obs_template
    obs = batch_concat_agent_id_to_obs(obs)

    N = int(agent.config['num_agents'])
    A = int(agent.config['action_dim'])
    D = N * A

    @jax.jit
    def q_critic(a_vec):
        actions_bt = a_vec.reshape(1, 1, N, A)
        obs_tb = switch_two_leading_dims(obs)
        actions_tb = switch_two_leading_dims(actions_bt)
        qs = agent.network.select('critic')(obs_tb, actions=actions_tb)
        q_mixed = qs.mean(axis=0).mean(axis=-1)[0, 0]
        return q_mixed

    @jax.jit
    def grad_norm(a_vec):
        g = jax.grad(q_critic)(a_vec)
        return jnp.linalg.norm(g)

    rng = jax.random.PRNGKey(seed)
    rng, sub = jax.random.split(rng)
    pts = jax.random.uniform(sub, (num_points, D), minval=-1.0, maxval=1.0)
    norms = jax.vmap(grad_norm)(pts)
    return float(jnp.max(norms))


# ============ Dataset builders ============ #

def make_dataset_quadratic(n_samples=50000, T=2, corr_sigma=0.15, seed: int = 0,
                           noise_p: float = 0.0, imbalance_q: float = 1.0,
                           coupling_c: float = 1.0):
    """
        corr_sigma: Gaussian jitter around the chosen manifold.
        noise_p: With prob p, sample actions uniformly from [-1,1]^2.
        imbalance_q: Mixture weight for diagonal (a2≈a1) vs anti-diagonal (a2≈-a1).
        coupling_c: Reward scaling factor.
    """
    rng = np.random.default_rng(seed)
    obs = np.zeros((n_samples, T, 2, 1), dtype=np.float32)
    acts = np.zeros((n_samples, T, 2, 1), dtype=np.float32)
    rews = np.zeros((n_samples, T, 2), dtype=np.float32)
    terms = np.zeros((n_samples, T, 2), dtype=np.float32)

    for i in range(n_samples):
        if rng.random() < noise_p:
            a1 = rng.uniform(-1.0, 1.0)
            a2 = rng.uniform(-1.0, 1.0)
        else:
            z = rng.uniform(-1.0, 1.0)
            if rng.random() < imbalance_q:
                # Diagonal cluster: a2 ≈ a1
                a1 = np.clip(z + rng.normal(0, corr_sigma), -1.0, 1.0)
                a2 = np.clip(z + rng.normal(0, corr_sigma), -1.0, 1.0)
            else:
                # Anti-diagonal cluster: a2 ≈ -a1
                a1 = np.clip(z + rng.normal(0, corr_sigma), -1.0, 1.0)
                a2 = np.clip(-z + rng.normal(0, corr_sigma), -1.0, 1.0)
        acts[i, 0, 0, 0] = a1
        acts[i, 0, 1, 0] = a2
        acts[i, 1] = acts[i, 0]
        r = coupling_c * float(q_env_quadratic(jnp.array([a1, a2])))
        rews[i, 0] = r
        rews[i, 1] = r
        terms[i, 1] = 1.0

    batch = {
        'observations': jnp.array(obs),
        'actions': jnp.array(acts),
        'rewards': jnp.array(rews),
        'terminals': jnp.array(terms),
    }
    return batch


def make_dataset_ring(n_samples=50000, T=2, noise_sigma=0.03, seed: int = 0,
                      noise_p: float = 0.0, imbalance_q: float = 0.5,
                      coupling_c: float = 1.0):
    """
        noise_sigma: Gaussian noise added to (cosθ, sinθ) coordinates.
        noise_p: With prob p, sample actions uniformly from [-1,1]^2.
        imbalance_q: Preference for same-sign quadrants (a1*a2>=0) vs opposite-sign (a1*a2<0).
        coupling_c: Reward scaling factor.
    """
    rng = np.random.default_rng(seed)
    obs = np.zeros((n_samples, T, 2, 1), dtype=np.float32)
    acts = np.zeros((n_samples, T, 2, 1), dtype=np.float32)
    rews = np.zeros((n_samples, T, 2), dtype=np.float32)
    terms = np.zeros((n_samples, T, 2), dtype=np.float32)

    for i in range(n_samples):
        if rng.random() < noise_p:
            a1 = rng.uniform(-1.0, 1.0)
            a2 = rng.uniform(-1.0, 1.0)
        else:
            theta = rng.uniform(0, 2 * np.pi)
            a1 = np.cos(theta)
            a2 = np.sin(theta)
            # Bias toward same-sign vs opposite-sign by rotating by π/2 when needed
            same_sign = (a1 * a2) >= 0
            if rng.random() < imbalance_q:
                # Prefer same-sign quadrants
                if not same_sign:
                    theta = (theta + np.pi / 2.0) % (2 * np.pi)
                    a1 = np.cos(theta)
                    a2 = np.sin(theta)
            else:
                # Prefer opposite-sign quadrants
                if same_sign:
                    theta = (theta + np.pi / 2.0) % (2 * np.pi)
                    a1 = np.cos(theta)
                    a2 = np.sin(theta)
            # Add small Gaussian noise and clip to [-1,1]
            a1 = float(np.clip(a1 + rng.normal(0, noise_sigma), -1.0, 1.0))
            a2 = float(np.clip(a2 + rng.normal(0, noise_sigma), -1.0, 1.0))
        acts[i, 0, 0, 0] = a1
        acts[i, 0, 1, 0] = a2
        acts[i, 1] = acts[i, 0]
        r = coupling_c * float(q_env_ring(jnp.array([a1, a2])))
        rews[i, 0] = r
        rews[i, 1] = r
        terms[i, 1] = 1.0

    batch = {
        'observations': jnp.array(obs),
        'actions': jnp.array(acts),
        'rewards': jnp.array(rews),
        'terminals': jnp.array(terms),
    }
    return batch


def make_dataset_xor(n_samples=50000, T=2, coupling_c: float = 1.0, noise_p: float = 0.0,
                     imbalance_q: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    B = n_samples
    N = 2
    obs = np.zeros((B, T, N, 1), dtype=np.float32)
    acts = np.zeros((B, T, N, 1), dtype=np.float32)
    rews = np.zeros((B, T, N), dtype=np.float32)
    terms = np.zeros((B, T, N), dtype=np.float32)

    def _xor_reward(a1: float, a2: float) -> float:
        return 1.0 if np.sign(a1) != np.sign(a2) else 0.0

    for i in range(B):
        if rng.random() < noise_p:
            a0 = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        else:
            if rng.random() < imbalance_q:
                a0 = np.array([+1, -1], dtype=np.float32)
            else:
                a0 = np.array([-1, +1], dtype=np.float32)
        acts[i, 0, :, 0] = a0
        acts[i, 1, :, 0] = a0

        r = coupling_c * _xor_reward(a0[0], a0[1])
        rews[i, 0] = r
        rews[i, 1] = r
        terms[i, 1] = 1.0

    batch = {
        'observations': jnp.array(obs),
        'actions': jnp.array(acts),
        'rewards': jnp.array(rews),
        'terminals': jnp.array(terms),
    }
    return batch


def make_dataset_landmark(n_episodes: int = 500,
                          T: int = 50,
                          seed: int = 0,
                          lm_positions: Optional[np.ndarray] = None,
                          start_sigma: float = 0.1,
                          start_mode: str = 'center',
                          lm_noise_sigma: float = 0.0,
                          lm_swap_q: float = 0.0,
                          lm_reward_c: float = 1.0,
                          lm_collision_lambda: float = 0.0,
                          lm_collision_radius: float = 0.15,
                          lm_speed_max: float = 0.02,
                          n_agents: int = 3,
                          lm_num_landmarks: Optional[int] = None,
                          lm_layout: str = 'auto',  # 'auto' | 'ring' | 'double_ring' | 'triple_ring' | 'twin_double_ring' | 'quad_double_ring' | 'triangle'
                          lm_radii: Optional[Sequence[float]] = None,
                          lm_counts: Optional[Sequence[int]] = None,
                          lm_include_center: bool = False,
                          lm_twin_center_sep: float = 1.0,
                          lm_twin_min_gap: float = 0.10,
                          lm_ring_band_min: float = 0.20,
                          lm_quad_center_sep: float = 1.0,
                          lm_quad_min_gap: float = 0.10,
                          reward_mode: str = 'auto') -> Dict[str, Any]:

    rng = np.random.default_rng(seed)
    N = int(n_agents)
    A = 2

    def _ring_points(k: int, radius: float, theta0: float = 0.0) -> np.ndarray:
        if k <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        th = np.linspace(0.0, 2 * np.pi, num=k, endpoint=False) + theta0
        return np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1).astype(np.float32)

    # Build landmark layout if not provided
    if lm_positions is None:
        # Determine total landmarks M
        if lm_counts is not None and len(lm_counts) > 0:
            M = int(sum(int(c) for c in lm_counts)) + (1 if lm_include_center else 0)
        elif lm_num_landmarks is not None and int(lm_num_landmarks) > 0:
            M = int(lm_num_landmarks)
        else:
            M = max(3, N)

        # Decide layout
        layout = (lm_layout or 'auto').lower()
        if layout == 'auto':
            if (M >= 80 or N >= 80):
                layout = 'quad_double_ring'
            elif (M >= 40 or N >= 40):
                layout = 'twin_double_ring'
            elif (M >= 20 or N >= 20):
                layout = 'double_ring'
            else:
                layout = 'ring' if M > 3 else 'triangle'

        pts = []
        if layout == 'triangle':
            base = np.array([[-1.0, 1.0], [0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
            if M <= 3:
                pts = base[:M]
            else:
                # Fill remaining uniformly on a ring
                rem = M - 3
                ring = _ring_points(rem, radius=1.2, theta0=0.0)
                pts = np.concatenate([base, ring], axis=0)
        elif layout == 'ring':
            r = float(lm_radii[0]) if (lm_radii and len(lm_radii) >= 1) else 1.0
            pts = _ring_points(M, radius=r, theta0=0.0)
        elif layout == 'double_ring':
            # Determine counts per ring
            if lm_counts and len(lm_counts) >= 2:
                c_in, c_out = int(lm_counts[0]), int(lm_counts[1])
                if lm_include_center:
                    # include a single center point
                    c_center = 1
                else:
                    c_center = 0
                total = c_center + c_in + c_out
                if total != M:
                    M = total
            else:
                c_in = max(6, int(round(0.4 * M)))
                c_out = max(6, M - c_in)
                c_center = 1 if lm_include_center else 0
                if c_center + c_in + c_out != M:
                    M = c_center + c_in + c_out
            r_in = float(lm_radii[0]) if (lm_radii and len(lm_radii) >= 1) else 0.6
            r_out = float(lm_radii[1]) if (lm_radii and len(lm_radii) >= 2) else 1.2
            rings = []
            if lm_include_center:
                rings.append(np.zeros((1, 2), dtype=np.float32))
            rings.append(_ring_points(c_in, r_in, theta0=0.0))
            rings.append(_ring_points(c_out, r_out, theta0=(np.pi / max(1, c_out))))
            pts = np.concatenate([p for p in rings if p.size > 0], axis=0)
        elif layout == 'triple_ring':
            # Determine counts per ring (inner, middle, outer)
            if lm_counts and len(lm_counts) >= 3:
                c_in, c_mid, c_out = int(lm_counts[0]), int(lm_counts[1]), int(lm_counts[2])
                c_center = 1 if lm_include_center else 0
                total = c_center + c_in + c_mid + c_out
                if total != M:
                    M = total
            else:
                # Default split roughly 0.3, 0.35, 0.35 ensuring >=6 each
                base = max(6, int(round(0.30 * M)))
                mid = max(6, int(round(0.35 * M)))
                rem = max(6, M - base - mid)
                c_in, c_mid, c_out = base, mid, rem
                c_center = 1 if lm_include_center else 0
                if c_center + c_in + c_mid + c_out != M:
                    M = c_center + c_in + c_mid + c_out
            # Radii defaults
            r_in = float(lm_radii[0]) if (lm_radii and len(lm_radii) >= 1) else 0.5
            r_mid = float(lm_radii[1]) if (lm_radii and len(lm_radii) >= 2) else 0.9
            r_out = float(lm_radii[2]) if (lm_radii and len(lm_radii) >= 3) else 1.3
            rings = []
            if lm_include_center:
                rings.append(np.zeros((1, 2), dtype=np.float32))
            rings.append(_ring_points(c_in, r_in, theta0=0.0))
            rings.append(_ring_points(c_mid, r_mid, theta0=(np.pi / max(1, c_mid))))
            rings.append(_ring_points(c_out, r_out, theta0=(np.pi / max(1, c_out))))
            pts = np.concatenate([p for p in rings if p.size > 0], axis=0)
        elif layout == 'twin_double_ring':
            # Two separate double-rings centered at c1 and c2.
            # Default centers at (-0.8, 0.0) and (+0.8, 0.0) unless overridden.
            sep = float(lm_twin_center_sep) if lm_twin_center_sep is not None else 0.8
            c1 = np.array([-sep, 0.0], dtype=np.float32)
            c2 = np.array([+sep, 0.0], dtype=np.float32)
            # Determine counts per (inner, outer) per side
            if lm_counts and len(lm_counts) >= 4:
                c_in_L, c_out_L, c_in_R, c_out_R = [int(x) for x in lm_counts[:4]]
                total = c_in_L + c_out_L + c_in_R + c_out_R + (2 if lm_include_center else 0)
                if total != M:
                    M = total
            else:
                # Split roughly equally across the two double-rings
                # First decide total per ring type
                c_in_total = max(6, int(round(0.4 * M / 2))) * 2
                c_out_total = max(6, int(round(0.6 * M / 2))) * 2
                # Adjust to match M (ignoring optional centers)
                base_total = c_in_total + c_out_total + (2 if lm_include_center else 0)
                if base_total != M:
                    # Distribute the difference to outer rings
                    diff = M - base_total
                    c_out_total = max(6, c_out_total + diff)
                c_in_L = c_in_total // 2
                c_in_R = c_in_total - c_in_L
                c_out_L = c_out_total // 2
                c_out_R = c_out_total - c_out_L
            # Radii per ring: start from base double_ring, but cap to avoid midline overlap.
            base_r_in = float(lm_radii[0]) if (lm_radii and len(lm_radii) >= 1) else 0.6
            base_r_out = float(lm_radii[1]) if (lm_radii and len(lm_radii) >= 2) else 1.2
            # Prevent overlap across x=0 (midline) by ensuring r_out <= sep - gap
            gap = float(lm_twin_min_gap)
            allowed_out = max(0.1, min(base_r_out, sep - gap))
            r_out = allowed_out
            r_in = max(0.05, min(base_r_in, r_out - float(lm_ring_band_min)))
            rings = []
            if lm_include_center:
                rings.append(c1[None, :])
                rings.append(c2[None, :])
            # Phase-offset outer to reduce overlap visually
            L_in = _ring_points(c_in_L, r_in, theta0=0.0) + c1
            L_out = _ring_points(c_out_L, r_out, theta0=(np.pi / max(1, c_out_L))) + c1
            R_in = _ring_points(c_in_R, r_in, theta0=0.0) + c2
            R_out = _ring_points(c_out_R, r_out, theta0=(np.pi / max(1, c_out_R))) + c2
            rings.extend([L_in, L_out, R_in, R_out])
            pts = np.concatenate([p for p in rings if p.size > 0], axis=0)
        elif layout == 'quad_double_ring':
            # Four separate double-rings centered in 4 quadrants.
            # Choose moderate radii and center separation to avoid boundary/axis overlap.
            sep = float(lm_quad_center_sep) if lm_quad_center_sep is not None else 0.75
            centers = [
                np.array([+sep, +sep], dtype=np.float32),  # Q1
                np.array([-sep, +sep], dtype=np.float32),  # Q2
                np.array([-sep, -sep], dtype=np.float32),  # Q3
                np.array([+sep, -sep], dtype=np.float32),  # Q4
            ]
            # Determine counts per (inner, outer) per quadrant
            if lm_counts and len(lm_counts) >= 8:
                counts = [int(x) for x in lm_counts[:8]]
                c_in = [counts[0], counts[2], counts[4], counts[6]]
                c_out = [counts[1], counts[3], counts[5], counts[7]]
                c_center = 4 if lm_include_center else 0
                total = sum(c_in) + sum(c_out) + c_center
                if total != M:
                    M = total
            else:
                # Split equally across 4 clusters
                c_center = 4 if lm_include_center else 0
                M_eff = max(0, M - c_center)
                per_cluster = max(12, M_eff // 4)
                c_in = []
                c_out = []
                for _ in range(4):
                    cin = max(6, int(round(0.40 * per_cluster)))
                    cout = max(6, per_cluster - cin)
                    c_in.append(cin)
                    c_out.append(cout)
                # Adjust for remainder
                used = sum(c_in) + sum(c_out)
                rem = M_eff - used
                i = 0
                while rem > 0:
                    c_out[i % 4] += 1
                    rem -= 1
                    i += 1
            # Radii tuned for clean 4-cluster view
            r_in = float(lm_radii[0]) if (lm_radii and len(lm_radii) >= 1) else 0.45
            r_out = float(lm_radii[1]) if (lm_radii and len(lm_radii) >= 2) else 0.85
            max_half = 1.48
            gap = float(lm_quad_min_gap)
            # Keep each ring inside its quadrant: prevent crossing axes
            r_out = min(r_out, max(0.1, sep - gap))
            # Ensure ring thickness
            if (r_out - r_in) < float(lm_ring_band_min):
                r_in = max(0.05, r_out - float(lm_ring_band_min))
            # Ensure bounds (keep centers, reduce r_out if needed)
            if (sep + r_out) > max_half:
                r_out = max(0.1, max_half - sep)
                if (r_out - r_in) < float(lm_ring_band_min):
                    r_in = max(0.05, r_out - float(lm_ring_band_min))

            rings = []
            if lm_include_center:
                rings.extend([c[None, :] for c in centers])
            # Use different phase offsets per ring to visually reduce overlaps
            phase = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
            for idx, c in enumerate(centers):
                th_in = 0.0
                th_out = phase[idx % len(phase)]
                Rin = _ring_points(c_in[idx], r_in, theta0=th_in) + c
                Rout = _ring_points(c_out[idx], r_out, theta0=th_out) + c
                rings.extend([Rin, Rout])
            pts = np.concatenate([p for p in rings if p.size > 0], axis=0)
        else:
            # Fallback: single ring
            pts = _ring_points(M, radius=1.0, theta0=0.0)
        lm_positions = pts.astype(np.float32)
    else:
        lm_positions = np.asarray(lm_positions, dtype=np.float32)
        assert lm_positions.ndim == 2 and lm_positions.shape[1] == 2, "lm_positions must be (M,2)"
    M = int(lm_positions.shape[0])

    obs_dim = 2 + 2 * M + (N - 1) * 2
    B = n_episodes
    obs = np.zeros((B, T, N, obs_dim), dtype=np.float32)
    acts = np.zeros((B, T, N, A), dtype=np.float32)
    rews = np.zeros((B, T, N), dtype=np.float32)
    terms = np.zeros((B, T, N), dtype=np.float32)

    for b in range(B):
        # Determine effective layout for starts (match landmark generation rules)
        layout_used = (lm_layout or 'auto').lower()
        if layout_used == 'auto':
            if (M >= 80 or N >= 80):
                layout_used = 'quad_double_ring'
            elif (M >= 40 or N >= 40):
                layout_used = 'twin_double_ring'
            elif (M >= 20 or N >= 20):
                layout_used = 'double_ring'
            else:
                layout_used = 'ring' if M > 3 else 'triangle'

        # Choose start mode
        effective_start_mode = (start_mode or 'auto').lower()
        if effective_start_mode == 'auto' and (N >= 100):
            # For many agents, default to centers implied by layout (no jitter)
            effective_start_mode = 'layout_centers'

        if effective_start_mode == 'center':
            starts = np.zeros((N, 2), dtype=np.float32)
            start_sigma_eff = 0.0
        elif effective_start_mode == 'quad_centers':
            sep = float(lm_quad_center_sep) if lm_quad_center_sep is not None else 1.0
            centers = [
                np.array([+sep, +sep], dtype=np.float32),
                np.array([-sep, +sep], dtype=np.float32),
                np.array([-sep, -sep], dtype=np.float32),
                np.array([+sep, -sep], dtype=np.float32),
            ]
            q = [N // 4, N // 4, N // 4, N // 4]
            for i in range(N - sum(q)):
                q[i % 4] += 1
            parts = [np.repeat(c[None, :], qk, axis=0) for c, qk in zip(centers, q)]
            starts = np.concatenate(parts, axis=0).astype(np.float32)
            start_sigma_eff = 0.0
        elif effective_start_mode == 'layout_centers':
            if layout_used == 'twin_double_ring':
                sep = float(lm_twin_center_sep) if lm_twin_center_sep is not None else 0.8
                c1 = np.array([-sep, 0.0], dtype=np.float32)
                c2 = np.array([+sep, 0.0], dtype=np.float32)
                nL = N // 2
                nR = N - nL
                starts_L = np.repeat(c1[None, :], nL, axis=0)
                starts_R = np.repeat(c2[None, :], nR, axis=0)
                starts = np.concatenate([starts_L, starts_R], axis=0).astype(np.float32)
            elif layout_used == 'quad_double_ring':
                sep = float(lm_quad_center_sep) if lm_quad_center_sep is not None else 1.0
                centers = [
                    np.array([+sep, +sep], dtype=np.float32),
                    np.array([-sep, +sep], dtype=np.float32),
                    np.array([-sep, -sep], dtype=np.float32),
                    np.array([+sep, -sep], dtype=np.float32),
                ]
                q = [N // 4, N // 4, N // 4, N // 4]
                for i in range(N - sum(q)):
                    q[i % 4] += 1
                parts = [np.repeat(c[None, :], qk, axis=0) for c, qk in zip(centers, q)]
                starts = np.concatenate(parts, axis=0).astype(np.float32)
            else:
                starts = np.zeros((N, 2), dtype=np.float32)
            start_sigma_eff = 0.0
        else:  # 'auto': aligned with number of circle clusters (double-ring counts as one)
            if layout_used == 'twin_double_ring':
                sep = float(lm_twin_center_sep) if lm_twin_center_sep is not None else 0.8
                c1 = np.array([-sep, 0.0], dtype=np.float32)
                c2 = np.array([+sep, 0.0], dtype=np.float32)
                nL = N // 2
                nR = N - nL
                starts_L = np.repeat(c1[None, :], nL, axis=0)
                starts_R = np.repeat(c2[None, :], nR, axis=0)
                starts = np.concatenate([starts_L, starts_R], axis=0).astype(np.float32)
            elif layout_used == 'quad_double_ring':
                sep = float(lm_quad_center_sep) if lm_quad_center_sep is not None else 1.0
                centers = [
                    np.array([+sep, +sep], dtype=np.float32),
                    np.array([-sep, +sep], dtype=np.float32),
                    np.array([-sep, -sep], dtype=np.float32),
                    np.array([+sep, -sep], dtype=np.float32),
                ]
                q = [N // 4, N // 4, N // 4, N // 4]
                for i in range(N - sum(q)):
                    q[i % 4] += 1
                parts = [np.repeat(c[None, :], qk, axis=0) for c, qk in zip(centers, q)]
                starts = np.concatenate(parts, axis=0).astype(np.float32)
            else:
                # Single-cluster layouts (ring/double_ring/triple_ring/triangle): spawn at the cluster center
                starts = np.zeros((N, 2), dtype=np.float32)
            start_sigma_eff = 0.0

        if start_sigma_eff and start_sigma_eff > 0.0:
            starts = starts + rng.normal(0.0, start_sigma_eff, size=starts.shape).astype(np.float32)
            starts = np.clip(starts, -1.5, 1.5)
        cost_start = np.stack([np.sum((starts[i][None, :] - lm_positions) ** 2, axis=1) for i in range(N)], axis=0)  # (N,M)
        assignment = _solve_assignment(cost_start)
        # Optional shuffle
        if rng.random() < lm_swap_q:
            rng.shuffle(assignment)
        # When layout uses multiple double-ring clusters, respawn at their known centers.
        if layout_used == 'twin_double_ring':
            # Exactly two centers on x-axis at ±sep; no quadrant splitting.
            sep = float(lm_twin_center_sep) if lm_twin_center_sep is not None else 0.8
            c1 = np.array([-sep, 0.0], dtype=np.float32)
            c2 = np.array([+sep, 0.0], dtype=np.float32)
            lm_x = lm_positions[assignment, 0]
            starts = np.stack([c2 if (lm_x[i] >= 0.0) else c1 for i in range(N)], axis=0)
            # Recompute assignment given new starts to keep target mapping consistent
            cost_start = np.stack([np.sum((starts[i][None, :] - lm_positions) ** 2, axis=1) for i in range(N)], axis=0)
            assignment = _solve_assignment(cost_start)
        elif layout_used == 'quad_double_ring':
            # Four centers at (±sep, ±sep); choose nearest center to the assigned landmark.
            sep = float(lm_quad_center_sep) if lm_quad_center_sep is not None else 1.0
            centers = np.array([
                [ +sep, +sep],
                [ -sep, +sep],
                [ -sep, -sep],
                [ +sep, -sep],
            ], dtype=np.float32)
            starts_list = []
            for i in range(N):
                j = int(assignment[i])
                diffs = centers - lm_positions[j]
                k = int(np.argmin(np.sum(diffs * diffs, axis=1)))
                starts_list.append(centers[k])
            starts = np.stack(starts_list, axis=0).astype(np.float32)
            # Recompute assignment given new starts to keep target mapping consistent
            cost_start = np.stack([np.sum((starts[i][None, :] - lm_positions) ** 2, axis=1) for i in range(N)], axis=0)
            assignment = _solve_assignment(cost_start)
        # Iterative environment stepping with max speed per step
        pos = starts.copy()  # (N,2)

        for t in range(T):
            # obs at current positions
            for i in range(N):
                self_pos = pos[i]
                other_pos = [pos[j] for j in range(N) if j != i]
                obs_vec = np.concatenate([self_pos,
                                          lm_positions.reshape(-1),
                                          np.concatenate(other_pos, axis=0) if other_pos else np.zeros((0,), dtype=np.float32)], axis=0)
                obs[b, t, i] = obs_vec
            # actions: capped step towards assigned landmark
            delta = np.zeros((N, 2), dtype=np.float32)
            for i in range(N):
                target = lm_positions[assignment[i]]
                dvec = target - pos[i]
                dist = float(np.linalg.norm(dvec))
                if dist < 1e-8:
                    step = np.zeros(2, dtype=np.float32)
                else:
                    step = (dvec / dist) * min(dist, lm_speed_max)
                delta[i] = step.astype(np.float32)
            acts[b, t] = delta

            # Rewards at current position: negative squared distance to optimal assignment
            # plus collision penalty (per-agent) when agents closer than radius.
            pos_t = pos  # (N,2)
            # Optimal assignment via DP (supports M>=N)
            cost_mat = np.stack([np.sum((pos_t[i][None, :] - lm_positions) ** 2, axis=1) for i in range(N)], axis=0)  # (N,M)
            assign = _solve_assignment(cost_mat)
            # Distance rewards by mode (optimal/greedy/fixed/nearest/auto)
            mode = (reward_mode or 'auto').lower()
            if mode == 'auto':
                # For large problems, use 'nearest' for speed
                mode = 'nearest' if (N >= 30 or M >= 30) else 'optimal'
            if mode in ('optimal', 'greedy'):
                for i in range(N):
                    d = pos_t[i] - lm_positions[assign[i]]
                    rews[b, t, i] = -lm_reward_c * float(np.dot(d, d))
            elif mode == 'fixed':
                # Use per-episode fixed assignment
                for i in range(N):
                    d = pos_t[i] - lm_positions[assignment[i]]
                    rews[b, t, i] = -lm_reward_c * float(np.dot(d, d))
            else:  # 'nearest'
                # Vectorized nearest landmark (no uniqueness constraint)
                diffs = pos_t[:, None, :] - lm_positions[None, :, :]  # (N,M,2)
                dist2 = np.sum(diffs * diffs, axis=-1)  # (N,M)
                min_dist2 = np.min(dist2, axis=1)  # (N,)
                rews[b, t, :] = -lm_reward_c * min_dist2
            # Collision penalties (symmetric)
            if lm_collision_lambda > 0.0:
                for i in range(N):
                    for j in range(i + 1, N):
                        if float(np.linalg.norm(pos_t[i] - pos_t[j])) < lm_collision_radius:
                            rews[b, t, i] -= lm_reward_c * lm_collision_lambda
                            rews[b, t, j] -= lm_reward_c * lm_collision_lambda

            # Environment update: apply step + optional positional noise
            pos = pos + delta
            if lm_noise_sigma > 0:
                pos = pos + rng.normal(0.0, lm_noise_sigma, size=pos.shape).astype(np.float32)

        terms[b, T - 1] = 1.0

    batch = {
        'observations': jnp.array(obs),
        'actions': jnp.array(acts),
        'rewards': jnp.array(rews),
        'terminals': jnp.array(terms),
        'landmarks': jnp.array(lm_positions),
    }
    return batch


# ============ Metrics ============ #

def _maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless-safe
        import matplotlib.pyplot as plt
        # Global style
        plt.rcParams["font.family"] = "monospace"
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except Exception:
        return None

def _save_two_versions(fig, base_path: str, title_text: Optional[str] = None, ensure_legend: bool = True):
    import matplotlib.pyplot as plt  # local import matches backend set

    axes = fig.axes if hasattr(fig, 'axes') else []

    # Ensure a legend exists if requested and handles present
    if ensure_legend:
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles and (ax.get_legend() is None):
                ax.legend(fontsize=9)

    # Apply title to the first axis (common case)
    if title_text and axes:
        axes[0].set_title(title_text)

    fig.tight_layout()
    fig.savefig(f"{base_path}_full.pdf")

    # Remove legends and titles for plain version
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                leg.set_visible(False)
        ax.set_title("")
    fig.tight_layout()
    fig.savefig(f"{base_path}_plain.pdf")

def _init_history() -> Dict[str, List[float]]:
    return {
        'step': [],
        'metric/env_lipschitz': [],
        'metric/critic_lipschitz': [],
        'metric/w2_proxy': [],
        'metric/sliced_w2': [],
        'metric/perf_joint': [],
        'metric/perf_fact': [],
        'metric/perf_gap': [],
        'metric/corr_joint': [],
        'metric/corr_fact': [],
        'metric/mi_joint': [],
        'metric/mi_fact': [],
        'actor/distill_loss': [],
    }

# --------- Utilities for landmark assignment --------- #
def _solve_assignment(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    N, M = cost.shape
    assert N <= M, "Currently supports N<=M"

    if M <= 12:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(i: int, mask: int) -> float:
            if i == N:
                return 0.0
            best = float('inf')
            for j in range(M):
                if not (mask & (1 << j)):
                    val = cost[i, j] + dp(i + 1, mask | (1 << j))
                    if val < best:
                        best = val
            return best

        assign = [-1] * N
        mask = 0
        for i in range(N):
            best = float('inf'); best_j = -1
            for j in range(M):
                if not (mask & (1 << j)):
                    val = cost[i, j] + dp(i + 1, mask | (1 << j))
                    if val < best:
                        best = val; best_j = j
            assign[i] = best_j
            mask |= (1 << best_j)
        return np.asarray(assign, dtype=np.int32)
    else:
        # Greedy: global sort all pairs by cost ascending, take first N disjoint pairs
        pairs = [(cost[i, j], i, j) for i in range(N) for j in range(M)]
        pairs.sort(key=lambda x: x[0])
        assign = [-1] * N
        used = set()
        for _, i, j in pairs:
            if assign[i] == -1 and j not in used:
                assign[i] = j
                used.add(j)
                if len(used) == N:
                    break
        # Fallback for any unassigned (shouldn't happen)
        if any(a == -1 for a in assign):
            free_cols = [j for j in range(M) if j not in used]
            for i in range(N):
                if assign[i] == -1:
                    assign[i] = free_cols.pop(0)
        return np.asarray(assign, dtype=np.int32)

def _update_history(history: Dict[str, List[float]], step: int, log_row: Dict[str, Any]):
    history['step'].append(int(step))
    for k in list(history.keys()):
        if k == 'step':
            continue
        history[k].append(float(log_row.get(k, float('nan'))))

def _save_curves(history: Dict[str, List[float]], out_dir: str, title_suffix: str = ""):
    plt = _maybe_import_matplotlib()
    if plt is None:
        # Matplotlib unavailable; skip plotting silently.
        return

    os.makedirs(out_dir, exist_ok=True)
    steps = history['step']
    if len(steps) == 0:
        return
    # Normalize steps to [0, 1]
    s0 = float(steps[0])
    sT = float(steps[-1]) if steps[-1] != steps[0] else (float(steps[-1]) + 1.0)
    steps_norm = (np.asarray(steps, dtype=float) - s0) / (sT - s0)

    # 1) Reward/performance curves (value metrics)
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, history['metric/perf_joint'], color=VALUE_COLOR, linestyle='solid', label='J_joint', marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, history['metric/perf_fact'], color=VALUE_COLOR, linestyle=':', label='J_factored', marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, history['metric/perf_gap'], color=VALUE_COLOR, linestyle='--', label='gap', marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Performance')
    plt.xticks([0.0, 0.5, 1.0])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'reward_curves'), title_text=f'Reward Curves{title_suffix}', ensure_legend=True)
    plt.close(fig)

    # 2) Distribution distance curves (policy metrics)
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, history['metric/w2_proxy'], color=POLICY_COLOR, linestyle='solid', label='W2 proxy', marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, history['metric/sliced_w2'], color=POLICY_COLOR, linestyle=':', label='SW2', marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Distance')
    plt.xticks([0.0, 0.5, 1.0])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'distance_curves'), title_text=f'Distribution Alignment{title_suffix}', ensure_legend=True)
    plt.close(fig)

    # 3) Lipschitz curves
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, history['metric/critic_lipschitz'], color=VALUE_COLOR, label='L_critic', marker='o', markerfacecolor='#FFFFFF')
    if not all([v == history['metric/env_lipschitz'][0] for v in history['metric/env_lipschitz']]):
        plt.plot(steps_norm, history['metric/env_lipschitz'], color=OTHER_COLOR, linestyle='--', label='L_env', marker='o', markerfacecolor='#FFFFFF')
    else:
        y = history['metric/env_lipschitz'][0]
        y_arr = np.full_like(steps_norm, y, dtype=float)
        plt.plot(steps_norm, y_arr, color=OTHER_COLOR, linestyle='--', label=f'L_env={y:.3f}', marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Lipschitz')
    plt.xticks([0.0, 0.5, 1.0])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'lipschitz_curves'), title_text=f'Lipschitz Tracking{title_suffix}', ensure_legend=True)
    plt.close(fig)

    # 4) Value gap (standalone small figure)
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, history['metric/perf_gap'], color=VALUE_COLOR, linestyle='--', label='value gap', marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Value gap')
    plt.xticks([0.0, 0.5, 1.0])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'value_gap_curve'), title_text=f'Value Gap{title_suffix}', ensure_legend=True)
    plt.close(fig)

    # 5) Dependency curves (Correlation)
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, history['metric/corr_joint'], color=POLICY_COLOR, linestyle='solid', label='corr (joint)', marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, history['metric/corr_fact'], color=POLICY_COLOR, linestyle=':', label='corr (factored)', marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Correlation')
    plt.xticks([0.0, 0.5, 1.0])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'dependency_corr'), title_text=f'Correlation{title_suffix}', ensure_legend=True)
    plt.close(fig)

    # 6) Dependency curves (Mutual Information)
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, history['metric/mi_joint'], color=POLICY_COLOR, linestyle='solid', label='MI (joint)', marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, history['metric/mi_fact'], color=POLICY_COLOR, linestyle=':', label='MI (factored)', marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Mutual information (hist)')
    plt.xticks([0.0, 0.5, 1.0])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'dependency_mi'), title_text=f'Mutual Information (hist){title_suffix}', ensure_legend=True)
    plt.close(fig)

def _save_theory_curves(history: Dict[str, List[float]], out_dir: str, title_suffix: str = ""):
    """Save curves that relate distillation loss, W2, value gap, and the Lipschitz bound.

    Produces:
      - distill_and_gap.png: distillation loss vs value gap (two y-axes)
      - w2_gap_bound.png: W2, value gap, and L*W2 overlayed
      - scatter_w2_vs_gap.png: scatter of (W2, |gap|) across steps with line y=L*x (L from last step)
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)

    steps = history.get('step', [])
    if not steps:
        return
    s0 = float(steps[0])
    sT = float(steps[-1]) if steps[-1] != steps[0] else (float(steps[-1]) + 1.0)
    steps_norm = (np.asarray(steps, dtype=float) - s0) / (sT - s0)

    distill = history.get('actor/distill_loss', [])
    w2 = history.get('metric/w2_proxy', [])
    gap = history.get('metric/perf_gap', [])
    Ls = history.get('metric/critic_lipschitz', [])

    # 1) Distillation vs Gap (two y-axes)
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 3), dpi=200)
    color1 = POLICY_COLOR
    color2 = VALUE_COLOR
    ax1.plot(steps_norm, distill, color=color1, label='distill loss', marker='o', markerfacecolor='#FFFFFF')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('distill loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xticks([0.0, 0.5, 1.0])
    ax2 = ax1.twinx()
    ax2.plot(steps_norm, gap, color=color2, label='value gap (joint - factored)', marker='o', markerfacecolor='#FFFFFF')
    ax2.set_ylabel('value gap', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    # Create a single legend combining both axes
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    if l1 or l2:
        ax1.legend(l1 + l2, lab1 + lab2, fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'distill_and_gap'), title_text=f'Distillation vs Value Gap{title_suffix}', ensure_legend=False)
    plt.close(fig)

    # 2) W2, Gap, and L*W2 overlay
    L_times_w2 = []
    for i in range(len(w2)):
        if i < len(Ls) and np.isfinite(Ls[i]):
            L_times_w2.append(Ls[i] * w2[i])
        else:
            L_times_w2.append(np.nan)
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(steps_norm, w2, label='W2 (proxy)', color=POLICY_COLOR, marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, np.abs(gap), label='|value gap|', color=VALUE_COLOR, marker='o', markerfacecolor='#FFFFFF')
    plt.plot(steps_norm, L_times_w2, label='L * W2 (bound)', color=OTHER_COLOR, marker='o', markerfacecolor='#FFFFFF')
    plt.xlabel('Steps')
    plt.ylabel('Magnitude')
    plt.xticks([0.0, 0.5, 1.0])
    plt.legend(fontsize=9)
    plt.grid(linestyle='--', alpha=0.3)
    _save_two_versions(fig, os.path.join(out_dir, 'w2_gap_bound'), title_text=f'W2 vs Value Gap with Bound{title_suffix}', ensure_legend=True)
    plt.close(fig)

    # 3) Scatter of W2 vs |gap| with y = L*x line using last L
    xs = np.asarray(w2, dtype=float)
    ys = np.abs(np.asarray(gap, dtype=float))
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.scatter(xs, ys, s=18, alpha=0.6, color=POLICY_COLOR)
    if len(Ls) > 0 and np.isfinite(Ls[-1]):
        L_last = float(Ls[-1])
        if np.isfinite(L_last) and np.isfinite(np.nanmax(xs)):
            x_line = np.linspace(0.0, float(np.nanmax(xs) * 1.05), 100)
            y_line = L_last * x_line
            plt.plot(x_line, y_line, linestyle='--', color=OTHER_COLOR, label=f'y = L*x (L={L_last:.2f})')
    plt.xlabel('W2 (proxy)')
    plt.ylabel('abs(value gap)')
    plt.legend(fontsize=9)
    plt.grid(linestyle='--', alpha=0.3)
    _save_two_versions(fig, os.path.join(out_dir, 'scatter_w2_vs_gap'), title_text=f'W2 vs Value Gap{title_suffix}', ensure_legend=True)
    plt.close(fig)

# ============ Dataset distribution plots ============ #
def _extract_actions(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return (a1, a2) arrays from dataset at t=0.

    Works for shapes (B, T, N, A) or (B, T, N).
    """
    acts = np.asarray(dataset['actions'])
    if acts.ndim == 3:  # (B, T, N)
        acts = acts[..., None]
    # Use first time step and first two agents
    a = acts[:, 0, :2]  # (B, 2, A)
    if a.shape[-1] == 1:
        a = a[..., 0]
    else:
        # If multi-dim actions, take the first component for plotting
        a = a[..., 0]
    a1 = a[:, 0]
    a2 = a[:, 1]
    return a1, a2


def _save_dataset_distribution(dataset, out_dir: str, toy: str = ""):
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    a1, a2 = _extract_actions(dataset)
    corr = float(np.corrcoef(a1, a2)[0, 1]) if a1.size > 1 else float('nan')

    # Joint distribution: hexbin in [-1, 1]^2
    fig = plt.figure(figsize=(4, 3), dpi=200)
    hb = plt.hexbin(a1, a2, gridsize=50, extent=(-1, 1, -1, 1), cmap='viridis', bins='log', mincnt=1)
    plt.colorbar(hb, label='log count')
    plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.xlabel('a1'); plt.ylabel('a2')
    # no title
    # Overlays per toy for intuition
    xs = np.linspace(-1, 1, 400)
    if toy == 'quadratic':
        plt.plot(xs, xs, 'w--', alpha=0.7, label='a1=a2')
        plt.legend(fontsize=9)
    if toy == 'ring':
        th = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(th), np.sin(th), 'w--', alpha=0.7, label='unit circle')
        plt.legend(fontsize=9)
    _save_two_versions(
        fig,
        os.path.join(out_dir, 'dataset_joint_distribution'),
        title_text=f'Dataset joint distribution {"[" + toy + "]" if toy else ""} (corr={corr:.3f})',
        ensure_legend=True,
    )
    plt.close(fig)


def _save_landmark_preview(dataset, out_dir: str, lm_positions: Optional[np.ndarray] = None):
    """Quick preview for the 3-agent landmark dataset.

    Plots final-step positions of agents across episodes and landmark locations.
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    obs = np.asarray(dataset['observations'])  # (B, T, N, 12)
    B, T, N, O = obs.shape
    # Decode positions from observations: first two entries are self pos
    final_self_pos = obs[:, -1, :, :2]  # (B, N, 2)
    # Support arbitrary N agents with a repeating color cycle
    base_colors = ['#BF5700', '#005F86', '#A29B96', '#8A8D8F', '#7D3C98', '#1F77B4', '#2CA02C', '#D62728']
    # Set global monospace font
    plt.rcParams['font.family'] = 'monospace'
    # Dynamic figure size: keep ring radius scale; widen canvas for multi-cluster
    # - twin (2 clusters): increase width only
    # - quad (4 clusters): increase both width and height
    # Default single-cluster size baseline
    base_w, base_h = 6, 4
    # Infer clusters from landmarks
    lm_for_cluster = np.array(dataset['landmarks']) if (lm_positions is None and 'landmarks' in dataset) else (
        np.array(lm_positions) if lm_positions is not None else _extract_landmarks_from_dataset(dataset)
    )
    nC = _infer_landmark_clusters(lm_for_cluster)['n'] if lm_for_cluster is not None else 1
    if nC == 2:
        figsize = (base_w * 2, base_h)
    elif nC >= 4:
        figsize = (base_w * 2, base_h * 2)
    else:
        figsize = (base_w, base_h)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    for i in range(N):
        ax.scatter(final_self_pos[:, i, 0], final_self_pos[:, i, 1], s=8, alpha=0.4,
                   color=base_colors[i % len(base_colors)], label=f'agent {i+1}')
    if lm_positions is None:
        if 'landmarks' in dataset:
            lm_positions = np.array(dataset['landmarks'])
        else:
            # Infer M from obs dim
            O = obs.shape[-1]
            N = obs.shape[2]
            M = int((O - 2 - 2*(N-1)) // 2)
            lm_positions = np.zeros((M, 2), dtype=np.float32)
    ax.scatter(lm_positions[:, 0], lm_positions[:, 1], marker='x', s=80, color='k', label='landmarks')
    # Widen x-axis based on landmarks and final positions
    try:
        max_abs_x = float(max(np.max(np.abs(lm_positions[:, 0])), np.max(np.abs(final_self_pos[..., 0]))))
        max_abs_y = float(max(np.max(np.abs(lm_positions[:, 1])), np.max(np.abs(final_self_pos[..., 1]))))
    except Exception:
        max_abs_x = float(np.max(np.abs(lm_positions[:, 0])))
        max_abs_y = float(np.max(np.abs(lm_positions[:, 1])))
    x_half = max(1.6, 1.1 * max_abs_x)
    # Keep vertical range independent to preserve ring radius for twin layouts
    y_half = max(1.6, 1.1 * max_abs_y)
    ax.set_xlim([-x_half, x_half]); ax.set_ylim([-y_half, y_half])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([round(-x_half, 1), 0.0, round(x_half, 1)]); ax.set_yticks([round(-y_half, 1), 0.0, round(y_half, 1)])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=9)
    # no title
    _save_two_versions(
        fig,
        os.path.join(out_dir, 'dataset_landmark_preview'),
        title_text='Landmark dataset: final positions',
        ensure_legend=True,
    )
    plt.close(fig)


def _extract_landmarks_from_dataset(dataset) -> np.ndarray:
    if 'landmarks' in dataset:
        return np.asarray(dataset['landmarks'])
    obs = np.asarray(dataset['observations'])
    O = obs.shape[-1]; N = obs.shape[2]
    M = int((O - 2 - 2*(N-1)) // 2)
    lm_flat = obs[0, 0, 0, 2:2+2*M]
    return lm_flat.reshape(M, 2)


def _group_landmarks_by_radius(landmarks: np.ndarray,
                               tol: float = 1e-3) -> Dict[str, Any]:
    """Group landmark indices into concentric rings by radius.

    Returns dict with keys:
      - 'center': indices near radius 0
      - 'rings': list of index lists per unique radius (sorted ascending)
      - 'radii': sorted unique radii values (non-center)
      - Convenience aliases when len(rings)>=3: 'inner', 'middle', 'outer'
    """
    pts = np.asarray(landmarks)
    r = np.linalg.norm(pts, axis=1)
    center_idx = np.where(r <= 0.1 + tol)[0].tolist()
    non_center_mask = r > 0.1 + tol
    r_nc = r[non_center_mask]
    idx_nc = np.where(non_center_mask)[0]
    if r_nc.size == 0:
        rings = []
        radii_vals = []
    else:
        # cluster by rounded radius to 3 decimals (consistent generation)
        radii_vals = np.unique(np.round(r_nc, 3)).tolist()
        radii_vals.sort()
        rings = []
        for rv in radii_vals:
            rings.append(idx_nc[np.where(np.abs(np.round(r_nc, 3) - rv) <= 0.0)[0]].tolist())
    out = {
        'center': center_idx,
        'rings': rings,
        'radii': radii_vals,
        'r_all': r,
    }
    if len(rings) >= 1:
        out['inner'] = rings[0]
    if len(rings) >= 2:
        out['outer'] = rings[-1]
    if len(rings) >= 3:
        out['middle'] = rings[1]
    return out

def _classify_endpoint_ring_idx(end_pos: np.ndarray,
                                landmarks: np.ndarray,
                                ring_groups: Dict[str, Any]) -> int:
    """Return ring index (0..R-1 sorted by radius) for endpoint's nearest landmark.

    Returns -1 for center if nearest is center.
    """
    diffs = landmarks - end_pos[None, :]
    j = int(np.argmin(np.sum(diffs * diffs, axis=1)))
    rings = ring_groups.get('rings', [])
    for ridx, idxs in enumerate(rings):
        if j in idxs:
            return ridx
    if j in ring_groups.get('center', []):
        return -1
    # Fallback by radius to closest ring
    rj = float(np.linalg.norm(landmarks[j]))
    radii = ring_groups.get('radii', [])
    if not radii:
        return 0
    diffs = [abs(rj - rv) for rv in radii]
    return int(np.argmin(diffs))


def _infer_landmark_clusters(landmarks: np.ndarray) -> Dict[str, Any]:
    """Infer coarse landmark clusters (1, 2, or 4) by position signs.

    Heuristics:
      - If both x and y strongly occupy positive and negative sides -> 4 quadrants
      - Else if x splits both sides -> 2 clusters (left/right)
      - Else -> 1 cluster (centered layout)

    Returns dict with keys:
      - 'labels': (M,) cluster id for each landmark
      - 'centers': list of (2,) centers per cluster id in order
      - 'n': number of clusters
    """
    pts = np.asarray(landmarks)
    xs = pts[:, 0]; ys = pts[:, 1]
    pos_x = float(np.mean(xs > 0.0)); neg_x = float(np.mean(xs < 0.0))
    pos_y = float(np.mean(ys > 0.0)); neg_y = float(np.mean(ys < 0.0))

    # Decide cluster count
    if pos_x > 0.2 and neg_x > 0.2 and pos_y > 0.2 and neg_y > 0.2:
        # 4 quadrants
        labels = []
        for x, y in zip(xs, ys):
            if x >= 0 and y >= 0:
                labels.append(0)
            elif x < 0 and y >= 0:
                labels.append(1)
            elif x < 0 and y < 0:
                labels.append(2)
            else:
                labels.append(3)
        labels = np.asarray(labels, dtype=np.int32)
        n = 4
    elif pos_x > 0.2 and neg_x > 0.2:
        # 2 clusters by x sign
        labels = (xs >= 0.0).astype(np.int32)
        n = 2
    else:
        labels = np.zeros((pts.shape[0],), dtype=np.int32)
        n = 1

    centers = []
    for c in range(n):
        sel = pts[labels == c]
        if sel.size == 0:
            centers.append(np.zeros(2, dtype=np.float32))
        else:
            centers.append(np.mean(sel, axis=0))
    return {'labels': labels, 'centers': centers, 'n': n}


def _save_landmark_trajectories_dataset_panels(dataset,
                                               out_dir: str,
                                               n_show: int = 30,
                                               max_trajectories: Optional[int] = None,
                                               t_stride: int = 1):
    """Panel view of dataset trajectories grouped by landmark clusters.

    Produces a small-multiples figure (1x2 or 2x2) for easier visual separation.
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    obs = np.asarray(dataset['observations'])
    landmarks = _extract_landmarks_from_dataset(dataset)
    B, T, N, F = obs.shape
    K = min(B, n_show)

    # Cluster landmarks and pre-classify endpoints by nearest landmark's cluster
    clusters = _infer_landmark_clusters(landmarks)
    lm_labels = clusters['labels']; nC = clusters['n']

    def nearest_lm_idx(p):
        d2 = np.sum((landmarks - p[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    end_pos = obs[:K, -1, :, :2]
    pair_cluster = {(b, i): int(lm_labels[nearest_lm_idx(end_pos[b, i])]) for b in range(K) for i in range(N)}

    # Selection: limit per agent across all clusters
    rng = np.random.default_rng(0)
    selected = set()
    if isinstance(max_trajectories, int) and max_trajectories > 0:
        per_agent_k = min(K, int(max_trajectories))
        for i in range(N):
            if per_agent_k >= K:
                for b in range(K):
                    selected.add((b, i))
            else:
                sel_b = rng.choice(K, size=per_agent_k, replace=False)
                for b in sel_b:
                    selected.add((int(b), i))
    else:
        for b in range(K):
            for i in range(N):
                selected.add((b, i))

    # Figure grid
    if nC == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=200)
        axes = [axes]
    elif nC == 2:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=200, sharex=False, sharey=False)
        axes = list(axes)
    else:  # 4
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=200, sharex=False, sharey=False)
        axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    base_colors = ['#BF5700', '#005F86', '#A29B96', '#8A8D8F', '#7D3C98', '#1F77B4', '#2CA02C', '#D62728']

    # Draw per cluster
    for c in range(nC):
        ax = axes[c]
        for b in range(K):
            pos_b = obs[b, :, :, :2]
            tidx = np.arange(0, pos_b.shape[0], max(1, t_stride))
            if tidx.size < 2:
                continue
            for i in range(N):
                if (b, i) not in selected or pair_cluster[(b, i)] != c:
                    continue
                xs = pos_b[tidx, i, 0]; ys = pos_b[tidx, i, 1]
                a0, a1 = 0.15, 0.9
                for k in range(len(tidx) - 1):
                    ak = a0 + (a1 - a0) * (k / max(1, len(tidx) - 2))
                    ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                            color=base_colors[i % len(base_colors)], alpha=ak, linewidth=1.0)
                ax.scatter([xs[0]], [ys[0]], s=18, facecolors='none',
                           edgecolors=base_colors[i % len(base_colors)], linewidths=0.8, alpha=0.5)
                ax.scatter([xs[-1]], [ys[-1]], s=28, facecolors='none',
                           edgecolors=base_colors[i % len(base_colors)], linewidths=1.1, alpha=0.95)
        # Landmarks in this cluster
        lm_sel = landmarks[lm_labels == c]
        ax.scatter(lm_sel[:, 0], lm_sel[:, 1], marker='x', s=70, color='k')
        # Axis bounds around points in this cluster
        if lm_sel.size > 0:
            minx = float(np.min(lm_sel[:, 0])); maxx = float(np.max(lm_sel[:, 0]))
            miny = float(np.min(lm_sel[:, 1])); maxy = float(np.max(lm_sel[:, 1]))
        else:
            minx = miny = -1.0; maxx = maxy = 1.0
        # Expand using any positions drawn (if any)
        # Collect points for cluster
        drawn = []
        for b in range(K):
            pos_b = obs[b, :, :, :2]
            for i in range(N):
                if (b, i) in selected and pair_cluster[(b, i)] == c:
                    drawn.append(pos_b[:, i, :])
        if drawn:
            P = np.concatenate(drawn, axis=0)
            minx = min(minx, float(np.min(P[:, 0]))); maxx = max(maxx, float(np.max(P[:, 0])))
            miny = min(miny, float(np.min(P[:, 1]))); maxy = max(maxy, float(np.max(P[:, 1])))
        cx = 0.5 * (minx + maxx); cy = 0.5 * (miny + maxy)
        half = max(maxx - minx, maxy - miny) * 0.6 + 0.25
        ax.set_xlim([cx - half, cx + half]); ax.set_ylim([cy - half, cy + half])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Cluster {c+1}')

    fig.tight_layout()
    _save_two_versions(fig, os.path.join(out_dir, 'dataset_landmark_panels'), title_text='Dataset trajectories (panels)', ensure_legend=False)
    plt.close(fig)


def _save_landmark_trajectories_policy_panels(agent: MACFlowAgent,
                                              dataset,
                                              out_dir: str,
                                              mode: str = 'joint',
                                              n_rollouts: int = 20,
                                              lm_speed_max: float = 0.02,
                                              max_trajectories: Optional[int] = None,
                                              t_stride: int = 1):
    """Panel view of policy rollouts grouped by inferred landmark clusters."""
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    obs = np.asarray(dataset['observations'])
    B, T, N, F = obs.shape
    K = min(B, n_rollouts)
    landmarks = _extract_landmarks_from_dataset(dataset)
    clusters = _infer_landmark_clusters(landmarks)
    lm_labels = clusters['labels']; nC = clusters['n']

    # Rollout and classify by nearest landmark's cluster
    rng = agent.rng
    trajs: List[np.ndarray] = []
    pair_cluster = {}
    for b in range(K):
        starts = obs[b, 0, :, :2]
        traj = _rollout_policy_landmark(agent, starts, landmarks, T=T, mode=mode, rng=rng, lm_speed_max=lm_speed_max)
        trajs.append(traj)
        for i in range(N):
            p = traj[i, -1]
            d2 = np.sum((landmarks - p[None, :]) ** 2, axis=1)
            j = int(np.argmin(d2))
            pair_cluster[(b, i)] = int(lm_labels[j])
        rng, _ = jax.random.split(rng)

    # Selection per agent
    rng_sel = np.random.default_rng(0)
    selected = set()
    if isinstance(max_trajectories, int) and max_trajectories > 0:
        per_agent_k = min(K, int(max_trajectories))
        for i in range(N):
            if per_agent_k >= K:
                for b in range(K):
                    selected.add((b, i))
            else:
                sel_b = rng_sel.choice(K, size=per_agent_k, replace=False)
                for b in sel_b:
                    selected.add((int(b), i))
    else:
        for b in range(K):
            for i in range(N):
                selected.add((b, i))

    # Figure grid
    if nC == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=200)
        axes = [axes]
    elif nC == 2:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=200, sharex=False, sharey=False)
        axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=200, sharex=False, sharey=False)
        axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    base_colors = ['#BF5700', '#005F86', '#A29B96', '#8A8D8F', '#7D3C98', '#1F77B4', '#2CA02C', '#D62728']

    for c in range(nC):
        ax = axes[c]
        # Draw trajectories
        for b in range(K):
            traj = trajs[b]
            tidx = np.arange(0, traj.shape[1], max(1, t_stride))
            if tidx.size < 2:
                continue
            for i in range(N):
                if (b, i) not in selected or pair_cluster[(b, i)] != c:
                    continue
                xs = traj[i, tidx, 0]; ys = traj[i, tidx, 1]
                a0, a1 = 0.15, 0.9
                for k in range(len(tidx) - 1):
                    ak = a0 + (a1 - a0) * (k / max(1, len(tidx) - 2))
                    ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                            color=base_colors[i % len(base_colors)], alpha=ak, linewidth=1.0)
                ax.scatter([xs[0]], [ys[0]], s=20, facecolors='none',
                           edgecolors=base_colors[i % len(base_colors)], linewidths=0.9, alpha=0.5)
                ax.scatter([xs[-1]], [ys[-1]], s=30, facecolors='none',
                           edgecolors=base_colors[i % len(base_colors)], linewidths=1.2, alpha=0.95)
        # Landmarks for this cluster
        lm_sel = landmarks[lm_labels == c]
        ax.scatter(lm_sel[:, 0], lm_sel[:, 1], marker='x', s=70, color='k')
        # Bounds
        if lm_sel.size > 0:
            minx = float(np.min(lm_sel[:, 0])); maxx = float(np.max(lm_sel[:, 0]))
            miny = float(np.min(lm_sel[:, 1])); maxy = float(np.max(lm_sel[:, 1]))
        else:
            minx = miny = -1.0; maxx = maxy = 1.0
        # Include drawn points
        drawn = []
        for b in range(K):
            for i in range(N):
                if (b, i) in selected and pair_cluster[(b, i)] == c:
                    drawn.append(trajs[b][i, :, :])
        if drawn:
            P = np.concatenate(drawn, axis=0)
            minx = min(minx, float(np.min(P[:, 0]))); maxx = max(maxx, float(np.max(P[:, 0])))
            miny = min(miny, float(np.min(P[:, 1]))); maxy = max(maxy, float(np.max(P[:, 1])))
        cx = 0.5 * (minx + maxx); cy = 0.5 * (miny + maxy)
        half = max(maxx - minx, maxy - miny) * 0.6 + 0.25
        ax.set_xlim([cx - half, cx + half]); ax.set_ylim([cy - half, cy + half])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Cluster {c+1}')

    fig.tight_layout()
    base = f'landmark_policy_panels_{mode}'
    _save_two_versions(fig, os.path.join(out_dir, base), title_text=f'Policy trajectories ({mode}, panels)', ensure_legend=False)
    plt.close(fig)


def _save_landmark_trajectories_dataset(dataset,
                                        out_dir: str,
                                        n_show: int = 30,
                                        style: str = 'scatter',
                                        max_trajectories: Optional[int] = None,
                                        t_stride: int = 1):
    """Plot multiple dataset trajectories (positions over time).

    Args:
      dataset: landmark dataset dict
      out_dir: directory to save figures
      n_show: number of episodes to plot
      style: 'scatter' (default) shows points with start/end markers,
             'line' saves a separate line-only plot.
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    obs = np.asarray(dataset['observations'])  # (B, T, N, F)
    B, T, N, F = obs.shape
    K = min(B, n_show)
    # Fixed dataset color palette (not policy-dependent)
    # Build a color cycle for arbitrary N
    base_colors = ['#BF5700', '#005F86', '#A29B96', '#8A8D8F', '#7D3C98', '#1F77B4', '#2CA02C', '#D62728']
    landmarks = _extract_landmarks_from_dataset(dataset)

    # Set global monospace font
    plt.rcParams['font.family'] = 'monospace'
    # Set global monospace font
    plt.rcParams['font.family'] = 'monospace'
    # Dynamic figure size based on cluster count (see preview logic)
    base_w, base_h = 6, 4
    landmarks = _extract_landmarks_from_dataset(dataset)
    nC = _infer_landmark_clusters(landmarks)['n'] if landmarks is not None else 1
    if nC == 2:
        figsize = (base_w * 2, base_h)
    elif nC >= 4:
        figsize = (base_w * 2, base_h * 2)
    else:
        figsize = (base_w, base_h)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    # Determine ring groups for layering
    ring_groups = _group_landmarks_by_radius(landmarks)
    # Pre-classify each episode/agent by target ring index using final positions
    end_pos = obs[:K, -1, :, :2]  # (K,N,2)
    labels = [[_classify_endpoint_ring_idx(end_pos[b, i], landmarks, ring_groups) for i in range(N)] for b in range(K)]
    R = max([rid for row in labels for rid in row if rid >= 0], default=-1) + 1
    ring_draw_order = list(range(R - 1, -1, -1))  # outer to inner

    # Build selection of episode-agent pairs to draw (limit per-agent)
    rng = np.random.default_rng(0)
    all_pairs = [(b, i) for b in range(K) for i in range(N)]
    if isinstance(max_trajectories, int) and max_trajectories > 0:
        selected = set()
        per_agent_k = min(K, int(max_trajectories))
        for i in range(N):
            if per_agent_k >= K:
                # Take all episodes for this agent
                for b in range(K):
                    selected.add((b, i))
            else:
                sel_b = rng.choice(K, size=per_agent_k, replace=False)
                for b in sel_b:
                    selected.add((int(b), i))
    else:
        selected = set(all_pairs)

    if style == 'line':
        # Draw rings by decreasing radius; inner last for visibility
        for ridx in ring_draw_order:
            for b in range(K):
                pos_b = obs[b, :, :, :2]
                for i in range(N):
                    if (b, i) not in selected or labels[b][i] != ridx:
                        continue
                    rel = 0 if R <= 1 else (ridx / max(1, R - 1))
                    z = 1 + rel
                    alpha_target = 0.55 + 0.35 * (1.0 - rel)
                    lw = 1.0 + 0.3 * (1.0 - rel)
                    tidx = np.arange(0, pos_b.shape[0], max(1, t_stride))
                    if tidx.size < 2:
                        continue
                    # Time-fading: early light -> late dark
                    a0 = max(0.05, 0.35 * alpha_target)
                    a1 = alpha_target
                    xs = pos_b[tidx, i, 0]
                    ys = pos_b[tidx, i, 1]
                    for k in range(len(tidx) - 1):
                        ak = a0 + (a1 - a0) * (k / max(1, len(tidx) - 2))
                        ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                                color=base_colors[i % len(base_colors)],
                                alpha=ak, linewidth=lw, zorder=z)
    else:
        # Scatter + line with time-fading per trajectory
        for ridx in ring_draw_order:
            for b in range(K):
                pos_b = obs[b, :, :, :2]
                for i in range(N):
                    if (b, i) not in selected or labels[b][i] != ridx:
                        continue
                    rel = 0 if R <= 1 else (ridx / max(1, R - 1))
                    z = 1 + rel
                    tidx = np.arange(0, pos_b.shape[0], max(1, t_stride))
                    if tidx.size < 2:
                        continue
                    a0 = 0.15
                    a1 = 0.85
                    xs = pos_b[tidx, i, 0]
                    ys = pos_b[tidx, i, 1]
                    for k in range(len(tidx) - 1):
                        ak = a0 + (a1 - a0) * (k / max(1, len(tidx) - 2))
                        ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                                color=base_colors[i % len(base_colors)],
                                alpha=ak, linewidth=1.0, zorder=z)
                    # Start/end markers
                    ax.scatter([xs[0]], [ys[0]], s=18, facecolors='none',
                               edgecolors=base_colors[i % len(base_colors)], linewidths=0.8,
                               alpha=0.5, zorder=z + 0.5)
                    ax.scatter([xs[-1]], [ys[-1]], s=28, facecolors='none',
                               edgecolors=base_colors[i % len(base_colors)], linewidths=1.1,
                               alpha=0.95, zorder=z + 0.6)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', s=90, color='k', label='landmarks', zorder=5)
    # Axis ranges: compute independently for x and y (keep ring size visually consistent)
    try:
        pos = obs[:K, :, :, :2]
        max_abs_x = float(max(np.max(np.abs(landmarks[:, 0])), np.max(np.abs(pos[..., 0]))))
        max_abs_y = float(max(np.max(np.abs(landmarks[:, 1])), np.max(np.abs(pos[..., 1]))))
    except Exception:
        max_abs_x = float(np.max(np.abs(landmarks[:, 0])))
        max_abs_y = float(np.max(np.abs(landmarks[:, 1])))
    x_half = max(1.6, 1.1 * max_abs_x)
    y_half = max(1.6, 1.1 * max_abs_y)
    ax.set_xlim([-x_half, x_half]); ax.set_ylim([-y_half, y_half])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([round(-x_half, 1), 0.0, round(x_half, 1)]); ax.set_yticks([round(-y_half, 1), 0.0, round(y_half, 1)])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    # no title
    title_str = 'Dataset trajectories (line)' if style == 'line' else 'Dataset trajectories (agents colored)'
    base = 'dataset_landmark_trajectories_line' if style == 'line' else 'dataset_landmark_trajectories'
    _save_two_versions(fig, os.path.join(out_dir, base), title_text=title_str, ensure_legend=False)
    plt.close(fig)


def _rollout_policy_landmark(agent: MACFlowAgent,
                             starts: np.ndarray,
                             landmarks: np.ndarray,
                             T: int,
                             mode: str = 'joint',
                             rng: Optional[Any] = None,
                             lm_speed_max: float = 0.02,
                             distance_aware_cap: bool = True) -> np.ndarray:
    """Rollout one trajectory of length T from given starts using the learned policy.

    Args:
      starts: (N, 2) starting positions
      landmarks: (3, 2)
      mode: 'joint' (flow) or 'factored' (one-step)
    Returns:
      traj: (N, T, 2)
    """
    N = starts.shape[0]
    A = int(agent.config['action_dim'])
    assert A == 2, 'landmark toy expects 2D actions'
    if rng is None:
        rng = agent.rng
    rng, sub = jax.random.split(rng)

    pos = starts.astype(np.float32).copy()
    traj = np.zeros((N, T, 2), dtype=np.float32)
    traj[:, 0] = pos
    for t in range(1, T):
        # Build vector observations (B=1,T=1)
        obs_vec = []
        for i in range(N):
            others = [pos[j] for j in range(N) if j != i]
            obs_i = np.concatenate([pos[i], landmarks.reshape(-1), np.concatenate(others, axis=0)], axis=0)
            obs_vec.append(obs_i)
        obs_vec = np.stack(obs_vec, axis=0)[None, None, ...]  # (1,1,N,F)
        # obs_with_ids = obs_vec
        obs_with_ids = batch_concat_agent_id_to_obs(jnp.asarray(obs_vec))

        if mode == 'joint' and hasattr(agent, 'compute_flow_actions'):
            rng, nkey = jax.random.split(rng)
            nse = jax.random.normal(nkey, (1, 1, N, A))
            acts = agent.compute_flow_actions(obs_with_ids, nse)[0, 0]
        else:
            rng, nkey = jax.random.split(rng)
            acts = agent.sample_actions(obs_with_ids, seed=nkey)[0, 0]
        a = np.array(acts)  # (N,2)
        # Cap step size. If distance_aware_cap, also avoid overshoot by limiting
        # each agent's step to its distance to the assigned landmark.
        norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
        if distance_aware_cap:
            # Compute optimal assignment to landmarks (supports general N,M)
            cost_mat = np.stack([np.sum((pos[i][None, :] - landmarks) ** 2, axis=1) for i in range(N)], axis=0)
            assign = _solve_assignment(cost_mat)
            dists = np.linalg.norm(pos - landmarks[assign], axis=1, keepdims=True)
            step_caps = np.minimum(lm_speed_max, dists)
            scale = np.minimum(1.0, step_caps / norms)
        else:
            scale = np.minimum(1.0, lm_speed_max / norms)
        a_capped = a * scale
        pos = pos + a_capped
        pos = np.clip(pos, -1.5, 1.5)
        traj[:, t] = pos
    return traj

def _landmark_step_rewards(pos_t: np.ndarray,
                           landmarks: np.ndarray,
                           lm_reward_c: float = 1.0,
                           lm_collision_lambda: float = 0.0,
                           lm_collision_radius: float = 0.15) -> np.ndarray:
    """Compute per-agent rewards at a single timestep for landmark toy.

    Uses optimal assignment (over 3! permutations) of agents to landmarks
    and negative squared distances, plus optional collision penalties.

    Args:
      pos_t: (N,2) agent positions at time t
      landmarks: (3,2) landmark positions
    Returns:
      rewards: (N,) per-agent rewards
    """
    N = pos_t.shape[0]
    M = landmarks.shape[0]
    # Build cost matrix and solve optimal assignment (M may be >= N)
    cost_mat = np.stack([np.sum((pos_t[i][None, :] - landmarks) ** 2, axis=1) for i in range(N)], axis=0)  # (N,M)
    assign = _solve_assignment(cost_mat)
    # Distance rewards per-agent
    rew = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        d = pos_t[i] - landmarks[assign[i]]
        rew[i] = -lm_reward_c * float(np.dot(d, d))
    # Collision penalties (symmetric)
    if lm_collision_lambda > 0.0:
        for i in range(N):
            for j in range(i + 1, N):
                if float(np.linalg.norm(pos_t[i] - pos_t[j])) < lm_collision_radius:
                    rew[i] -= lm_reward_c * lm_collision_lambda
                    rew[j] -= lm_reward_c * lm_collision_lambda
    return rew

def _estimate_landmark_perf(agent: MACFlowAgent,
                            dataset,
                            mode: str,
                            n_rollouts: int = 10,
                            lm_speed_max: float = 0.02,
                            lm_reward_c: float = 1.0,
                            lm_collision_lambda: float = 0.0,
                            lm_collision_radius: float = 0.15) -> float:
    """Estimate landmark policy performance by rollouts with environment reward.

    Returns average per-step per-agent reward across episodes.
    """
    obs = np.asarray(dataset['observations'])
    # Extract landmarks for rollouts
    landmarks = _extract_landmarks_from_dataset(dataset)
    B, T, N, F = obs.shape
    K = min(B, n_rollouts)
    # landmarks already extracted above
    rng = agent.rng
    acc = 0.0
    count = 0
    for b in range(K):
        starts = obs[b, 0, :, :2]
        traj = _rollout_policy_landmark(agent, starts, landmarks, T=T, mode=mode, rng=rng, lm_speed_max=lm_speed_max)
        # Per-step rewards
        for t in range(T):
            r_t = _landmark_step_rewards(traj[:, t], landmarks,
                                         lm_reward_c=lm_reward_c,
                                         lm_collision_lambda=lm_collision_lambda,
                                         lm_collision_radius=lm_collision_radius)
            acc += float(np.mean(r_t))  # average per-agent
            count += 1
        rng, _ = jax.random.split(rng)
    return acc / max(count, 1)


def _save_landmark_trajectories_policy(agent: MACFlowAgent,
                                       dataset,
                                       out_dir: str,
                                       mode: str = 'joint',
                                       n_rollouts: int = 20,
                                       lm_speed_max: float = 0.02,
                                       style: str = 'scatter',
                                       max_trajectories: Optional[int] = None,
                                       t_stride: int = 1):
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    obs = np.asarray(dataset['observations'])
    B, T, N, F = obs.shape
    K = min(B, n_rollouts)
    # Color palette per agent; supports arbitrary N
    base_colors = ['#BF5700', '#005F86', '#A29B96', '#8A8D8F', '#7D3C98', '#1F77B4', '#2CA02C', '#D62728']
    landmarks = _extract_landmarks_from_dataset(dataset)

    # Dynamic figure size: base size for single cluster, widen for twin, widen+heighten for quad
    base_w, base_h = 4, 3
    nC = _infer_landmark_clusters(landmarks)['n'] if landmarks is not None else 1
    if nC == 2:
        figsize = (base_w * 2, base_h)
    elif nC >= 4:
        figsize = (base_w * 2, base_h * 2)
    else:
        figsize = (base_w, base_h)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    rng = agent.rng
    ring_groups = _group_landmarks_by_radius(landmarks)
    # Pre-roll all trajectories to classify endpoints, then draw by ring
    trajs: List[np.ndarray] = []  # list of (N,T,2)
    labels: List[List[int]] = []  # per rollout per agent ring index
    for b in range(K):
        starts = obs[b, 0, :, :2]
        traj = _rollout_policy_landmark(agent, starts, landmarks, T=T, mode=mode, rng=rng, lm_speed_max=lm_speed_max)
        trajs.append(traj)
        labels.append([_classify_endpoint_ring_idx(traj[i, -1], landmarks, ring_groups) for i in range(N)])
        rng, _ = jax.random.split(rng)

    # Build selection of rollout-agent pairs to draw (limit per-agent)
    rng_sel = np.random.default_rng(0)
    all_pairs = [(b, i) for b in range(K) for i in range(N)]
    if isinstance(max_trajectories, int) and max_trajectories > 0:
        selected = set()
        per_agent_k = min(K, int(max_trajectories))
        for i in range(N):
            if per_agent_k >= K:
                for b in range(K):
                    selected.add((b, i))
            else:
                sel_b = rng_sel.choice(K, size=per_agent_k, replace=False)
                for b in sel_b:
                    selected.add((int(b), i))
    else:
        selected = set(all_pairs)

    # Determine number of rings and draw order (outer to inner)
    R = max([rid for row in labels for rid in row if rid >= 0], default=-1) + 1
    ring_draw_order = list(range(R - 1, -1, -1))
    for ridx in ring_draw_order:
        for b in range(K):
            traj = trajs[b]
            for i in range(N):
                if (b, i) not in selected or labels[b][i] != ridx:
                    continue
                rel = 0 if R <= 1 else (ridx / max(1, R - 1))
                z = 1 + rel
                tidx = np.arange(0, traj.shape[1], max(1, t_stride))
                if tidx.size < 2:
                    continue
                xs = traj[i, tidx, 0]
                ys = traj[i, tidx, 1]
                if style == 'line':
                    a1 = (0.55 + 0.35 * (1.0 - rel))
                    a0 = max(0.05, 0.35 * a1)
                    lw = (1.0 + 0.3 * (1.0 - rel))
                    for k in range(len(tidx) - 1):
                        ak = a0 + (a1 - a0) * (k / max(1, len(tidx) - 2))
                        ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                                color=base_colors[i % len(base_colors)],
                                alpha=ak, linewidth=lw, zorder=z)
                else:
                    a0 = 0.15
                    a1 = 0.9
                    for k in range(len(tidx) - 1):
                        ak = a0 + (a1 - a0) * (k / max(1, len(tidx) - 2))
                        ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                                color=base_colors[i % len(base_colors)],
                                alpha=ak, linewidth=1.0, zorder=z)
                    ax.scatter([xs[0]], [ys[0]], s=22, facecolors='none',
                               edgecolors=base_colors[i % len(base_colors)], linewidths=0.9,
                               alpha=0.5, zorder=z + 0.5)
                    ax.scatter([xs[-1]], [ys[-1]], s=32, facecolors='none',
                               edgecolors=base_colors[i % len(base_colors)], linewidths=1.2,
                               alpha=0.95, zorder=z + 0.6)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', s=90, color='k', label='landmarks', zorder=5)
    # Widen x-axis dynamically based on landmarks and generated trajectories
    try:
        # Concatenate all rolled trajectories for axis ranges
        traj_concat = np.concatenate(trajs, axis=0) if len(trajs) > 0 else None
        max_abs_x = float(np.max(np.abs(traj_concat[..., 0]))) if traj_concat is not None else 0.0
        max_abs_y = float(np.max(np.abs(traj_concat[..., 1]))) if traj_concat is not None else 0.0
        max_abs_x = float(max(max_abs_x, np.max(np.abs(landmarks[:, 0]))))
        max_abs_y = float(max(max_abs_y, np.max(np.abs(landmarks[:, 1]))))
    except Exception:
        max_abs_x = float(np.max(np.abs(landmarks[:, 0])))
        max_abs_y = float(np.max(np.abs(landmarks[:, 1])))
    x_half = max(1.6, 1.1 * max_abs_x)
    y_half = max(1.6, 1.1 * max_abs_y)
    ax.set_xlim([-x_half, x_half]); ax.set_ylim([-y_half, y_half])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([round(-x_half, 1), 0.0, round(x_half, 1)]); ax.set_yticks([round(-y_half, 1), 0.0, round(y_half, 1)])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    # no title
    title = f'Policy trajectories ({mode}, line)' if style == 'line' else f'Policy trajectories ({mode})'
    base = f'landmark_policy_trajectories_{mode}_line' if style == 'line' else f'landmark_policy_trajectories_{mode}'
    _save_two_versions(fig, os.path.join(out_dir, base), title_text=title, ensure_legend=False)
    plt.close(fig)


def _extract_a1_a2_from_policy_samples(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize policy samples to (S,) a1 and a2 for plotting.

    Accepts shapes:
      - (S, 2): scalar action per agent
      - (S, 2, A): multi-dim action; we take the first component
    """
    X = np.asarray(samples)
    if X.ndim == 2 and X.shape[1] == 2:
        a1 = X[:, 0]
        a2 = X[:, 1]
    elif X.ndim == 3 and X.shape[1] == 2:
        a1 = X[:, 0, 0]
        a2 = X[:, 1, 0]
    else:
        # Fallback: attempt to squeeze and take first two dims
        Xs = np.squeeze(X)
        if Xs.ndim == 1 and Xs.shape[0] == 2:
            a1 = np.array([Xs[0]])
            a2 = np.array([Xs[1]])
        elif Xs.ndim >= 2:
            a1 = Xs[:, 0]
            a2 = Xs[:, 1]
        else:
            a1 = np.asarray([])
            a2 = np.asarray([])
    return a1, a2


def _save_policy_distribution(samples: np.ndarray, out_dir: str, name: str, toy: str = ""):
    """Save joint hexbin and marginals for a policy's action samples.

    Args:
      samples: np.ndarray of shape (S, 2) or (S, 2, A)
      out_dir: directory to save figures
      name: identifier in filenames, e.g., 'joint' or 'factored'
      toy: optional toy name for overlays
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    a1, a2 = _extract_a1_a2_from_policy_samples(samples)
    if a1.size == 0 or a2.size == 0:
        return
    corr = float(np.corrcoef(a1, a2)[0, 1]) if a1.size > 1 else float('nan')

    # Joint distribution
    fig = plt.figure(figsize=(3, 2.5), dpi=200)
    hb = plt.hexbin(a1, a2, gridsize=50, extent=(-1, 1, -1, 1), cmap='viridis', bins='log', mincnt=1)
    plt.colorbar(hb, label='log count')
    plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.xlabel('a1'); plt.ylabel('a2')
    # no title
    xs = np.linspace(-1, 1, 400)
    if toy == 'quadratic':
        plt.plot(xs, xs, 'w--', alpha=0.7, label='a1=a2')
        plt.legend(fontsize=9)
    if toy == 'ring':
        th = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(th), np.sin(th), 'w--', alpha=0.7, label='unit circle')
        plt.legend(fontsize=9)
    _save_two_versions(
        fig,
        os.path.join(out_dir, f'policy_{name}_joint'),
        title_text=f'Policy {name} joint {"[" + toy + "]" if toy else ""} (corr={corr:.3f})',
        ensure_legend=True,
    )
    plt.close(fig)

    # Marginals
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=200, sharex=True, sharey=True)
    bins = np.linspace(-1, 1, 51)
    axes[0].hist(a1, bins=bins, color=POLICY_COLOR, alpha=0.8, density=True)
    # no title
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(a2, bins=bins, color=POLICY_COLOR, alpha=0.8, density=True)
    # no title
    for ax in axes:
        ax.set_xlim([-1, 1])
        ax.set_xlabel('action'); ax.set_ylabel('density')
    _save_two_versions(
        fig,
        os.path.join(out_dir, f'policy_{name}_marginals'),
        title_text=f'Policy {name} marginals {"[" + toy + "]" if toy else ""}',
        ensure_legend=False,
    )
    plt.close(fig)


def _save_policy_scatter_density(samples: np.ndarray, out_dir: str, name: str, toy: str = ""):
    """Save scatter + density contour for a policy's samples.

    Args:
      samples: np.ndarray of shape (S, 2) or (S, 2, A)
      out_dir: directory to save figures
      name: identifier in filenames, e.g., 'joint' or 'factored'
      toy: optional toy name for overlays
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    a1, a2 = _extract_a1_a2_from_policy_samples(samples)
    if a1.size == 0 or a2.size == 0:
        return
    corr = float(np.corrcoef(a1, a2)[0, 1]) if a1.size > 1 else float('nan')

    # Scatter + histogram-smoothed contour in [-1, 1]^2
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=200)
    ax.scatter(a1, a2, s=3, alpha=0.25, color=POLICY_COLOR)
    bins = 128
    H, xedges, yedges = np.histogram2d(a1, a2, bins=bins, range=[[-1, 1], [-1, 1]], density=True)

    def _gauss1d(sigma=1.0, radius=3):
        xs = np.arange(-radius, radius + 1)
        k = np.exp(-0.5 * (xs / sigma) ** 2)
        k /= np.sum(k)
        return k
    def _smooth2d(img, sigma=1.0, radius=2):
        k = _gauss1d(sigma=sigma, radius=radius)
        tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=1, arr=img)
        sm = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=0, arr=tmp)
        return sm

    Hs = _smooth2d(H, sigma=1.0, radius=2)
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    CS = ax.contour(Xc, Yc, Hs.T, levels=8, colors='k', linewidths=0.7, alpha=0.85)
    ax.clabel(CS, inline=True, fontsize=7, fmt='%.2f')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ax.set_xlabel('a1'); ax.set_ylabel('a2')
    # no title

    xs = np.linspace(-1, 1, 400)
    if toy == 'quadratic':
        ax.plot(xs, xs, 'r--', alpha=0.6, linewidth=1.0, label='a1=a2')
        ax.legend(fontsize=9)
    if toy == 'ring':
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), 'r--', alpha=0.6, linewidth=1.0, label='unit circle')
        ax.legend(fontsize=9)

    _save_two_versions(
        fig,
        os.path.join(out_dir, f'policy_{name}_scatter_density'),
        title_text=f'Policy {name} scatter + density {"[" + toy + "]" if toy else ""} (corr={corr:.3f})',
        ensure_legend=True,
    )
    plt.close(fig)

def _compute_q_grid_critic(agent: MACFlowAgent,
                           obs_template: Optional[jnp.ndarray] = None,
                           grid_size: int = 81):
    """Compute learned critic Q grid over [-1,1]^2 for 2 agents, 1D actions.

    Returns (a1s, a2s, q_grid, idx_joint, (i_fact, j_fact)).
    """
    if obs_template is None:
        obs = jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)
    else:
        obs = obs_template
    obs = batch_concat_agent_id_to_obs(obs)
    obs_np = np.array(obs)

    a1s = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    a2s = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    A1, A2 = np.meshgrid(a1s, a2s, indexing='ij')
    B = grid_size * grid_size

    actions = np.zeros((B, 1, 2, 1), dtype=np.float32)
    actions[:, 0, 0, 0] = A1.reshape(-1)
    actions[:, 0, 1, 0] = A2.reshape(-1)
    obs_batch = np.repeat(obs_np, repeats=B, axis=0)

    qs = agent.network.select('critic')(jnp.asarray(obs_batch), actions=jnp.asarray(actions))
    qs_np = np.array(qs)
    q_mixed = qs_np.mean(axis=0).mean(axis=-1)  # (B,1)
    q_flat = q_mixed.squeeze(-1)  # (B,)
    q_grid = q_flat.reshape((grid_size, grid_size))

    # Joint argmax
    idx_joint = np.unravel_index(np.argmax(q_grid), q_grid.shape)
    # Factorized: maximize marginals
    q1 = q_grid.max(axis=1)
    q2 = q_grid.max(axis=0)
    i_fact = int(np.argmax(q1))
    j_fact = int(np.argmax(q2))
    return a1s, a2s, q_grid, idx_joint, (i_fact, j_fact)

def _save_xor_q_heatmap(agent: MACFlowAgent,
                        out_dir: str,
                        obs_template: Optional[jnp.ndarray] = None,
                        grid_size: int = 81,
                        use_env: bool = False,
                        q_env_jax=None):
    """Save Q heatmap for XOR: learned critic if available else env Q.

    Produces full/plain PDFs with joint and factored argmax markers.
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)

    if not use_env:
        a1s, a2s, q_grid, idx_joint, (i_fact, j_fact) = _compute_q_grid_critic(agent, obs_template=obs_template, grid_size=grid_size)
        title = 'Q_tot heatmap (critic)'
    else:
        a1s, a2s, q_grid = compute_q_grid_env(q_env_jax, grid_size=grid_size)
        idx_joint = np.unravel_index(np.argmax(q_grid), q_grid.shape)
        q1 = q_grid.max(axis=1)
        q2 = q_grid.max(axis=0)
        i_fact = int(np.argmax(q1))
        j_fact = int(np.argmax(q2))
        title = 'Q_env heatmap'

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.8), dpi=200)
    im = ax.imshow(q_grid.T, origin='lower', extent=[a1s[0], a1s[-1], a2s[0], a2s[-1]], aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Q')
    a1_joint, a2_joint = a1s[idx_joint[0]], a2s[idx_joint[1]]
    a1_fact, a2_fact = a1s[i_fact], a2s[j_fact]
    ax.scatter([a1_joint], [a2_joint], c='red', marker='x', s=80, label='joint argmax')
    ax.scatter([a1_fact], [a2_fact], c='white', marker='o', s=40, label='factored argmax')
    ax.set_xlabel('a1'); ax.set_ylabel('a2'); ax.legend(loc='upper right', fontsize=9)
    _save_two_versions(fig, os.path.join(out_dir, 'q_heatmap'), title_text=title, ensure_legend=True)
    plt.close(fig)

def _save_xor_samples_figure(dataset_actions, joint_samples: np.ndarray, factored_samples: np.ndarray, out_dir: str):
    """Save a 3-panel scatter: dataset, joint-flow, factored actions (XOR only)."""
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)

    da = np.array(dataset_actions)
    # dataset may be (B,T,N,1) or (B,T,N)
    if da.ndim == 4:
        da2 = da[:, 0, :, 0]
    elif da.ndim == 3:
        da2 = da[:, 0, :]
    else:
        da2 = da.reshape(da.shape[0], -1)[:, :2]
    # Subsample for readability
    try:
        n = da2.shape[0]
        k = min(n, 8000)
        sel = np.random.choice(n, size=k, replace=False)
        da_plot = da2[sel]
    except Exception:
        da_plot = da2

    # Normalize samples to (S,) a1 and a2 using existing helper
    a1j, a2j = _extract_a1_a2_from_policy_samples(joint_samples)
    a1f, a2f = _extract_a1_a2_from_policy_samples(factored_samples)

    fig = plt.figure(figsize=(12, 3), dpi=200)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    ax0.scatter(da_plot[:, 0], da_plot[:, 1], s=6, alpha=0.5, color='#A6A6A6', label='dataset')
    ax0.set_xlabel('a1'); ax0.set_ylabel('a2')
    ax0.set_xlim([-1.2, 1.2]); ax0.set_ylim([-1.2, 1.2]); ax0.grid(True, alpha=0.2)
    ax0.legend(fontsize=9)

    ax1.scatter(a1j, a2j, s=6, alpha=0.6, color=POLICY_COLOR, label='joint')
    ax1.set_xlabel('a1'); ax1.set_ylabel('a2')
    ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([-1.2, 1.2]); ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=9)

    ax2.scatter(a1f, a2f, s=6, alpha=0.6, color='#595959', label='factored')
    ax2.set_xlabel('a1'); ax2.set_ylabel('a2')
    ax2.set_xlim([-1.2, 1.2]); ax2.set_ylim([-1.2, 1.2]); ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save_two_versions(fig, os.path.join(out_dir, 'samples_scatter'), title_text='XOR action distributions', ensure_legend=True)
    plt.close(fig)

def _save_xor_samples_figure(dataset_actions, joint_samples: np.ndarray, factored_samples: np.ndarray, out_dir: str):
    """Save a 3-panel scatter: dataset, joint-flow, factored actions (XOR only).

    Saves to samples_scatter_full.pdf and samples_scatter_plain.pdf in out_dir.
    """
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)

    da = np.array(dataset_actions)
    # dataset may be (B,T,N,1) or (B,T,N)
    if da.ndim == 4:
        da2 = da[:, 0, :, 0]
    elif da.ndim == 3:
        da2 = da[:, 0, :]
    else:
        da2 = da.reshape(da.shape[0], -1)[:, :2]
    # Subsample for readability
    try:
        n = da2.shape[0]
        k = min(n, 8000)
        sel = np.random.choice(n, size=k, replace=False)
        da_plot = da2[sel]
    except Exception:
        da_plot = da2

    # Extract coordinates robustly (handles (S,2), (S,2,1), etc.)
    a1j, a2j = _extract_a1_a2_from_policy_samples(joint_samples)
    a1f, a2f = _extract_a1_a2_from_policy_samples(factored_samples)

    fig = plt.figure(figsize=(12, 3), dpi=200)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    ax0.scatter(da_plot[:, 0], da_plot[:, 1], s=6, alpha=0.5, color='#A6A6A6', label='dataset')
    ax0.set_xlabel('a1'); ax0.set_ylabel('a2')
    ax0.set_xlim([-1.2, 1.2]); ax0.set_ylim([-1.2, 1.2]); ax0.grid(True, alpha=0.2)
    ax0.legend(fontsize=9)

    ax1.scatter(a1j, a2j, s=6, alpha=0.6, color='#BF5700', label='joint')
    ax1.set_xlabel('a1'); ax1.set_ylabel('a2')
    ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([-1.2, 1.2]); ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=9)

    ax2.scatter(a1f, a2f, s=6, alpha=0.6, color='#595959', label='factored')
    ax2.set_xlabel('a1'); ax2.set_ylabel('a2')
    ax2.set_xlim([-1.2, 1.2]); ax2.set_ylim([-1.2, 1.2]); ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save_two_versions(fig, os.path.join(out_dir, 'samples_scatter'), title_text='XOR action distributions', ensure_legend=True)
    plt.close(fig)

def sample_joint_actions(agent: MACFlowAgent, n: int, rng, obs_template: Optional[jnp.ndarray] = None) -> np.ndarray:
    """Sample n joint-flow actions; return shape (n, N*A).

    If obs_template is provided, it must have shape (1, 1, N, O).
    """
    if obs_template is None:
        N = int(agent.config['num_agents'])
        obs = jnp.zeros((1, 1, N, 1), dtype=jnp.float32)
    else:
        obs = obs_template
    obs = batch_concat_agent_id_to_obs(obs)
    a_list = []
    r = rng
    N = int(agent.config['num_agents'])
    A = int(agent.config['action_dim'])
    for _ in range(n):
        r, sub = jax.random.split(r)
        nse = jax.random.normal(sub, (1, 1, N, A))
        acts = agent.compute_flow_actions(obs, nse)[0, 0]  # (N, A)
        a_list.append(np.array(acts).reshape(-1))
    return np.asarray(a_list)


def sample_factored_actions(agent: MACFlowAgent, n: int, rng, obs_template: Optional[jnp.ndarray] = None) -> np.ndarray:
    if obs_template is None:
        N = int(agent.config['num_agents'])
        obs = jnp.zeros((1, 1, N, 1), dtype=jnp.float32)
    else:
        obs = obs_template
    obs = batch_concat_agent_id_to_obs(obs)
    a_list = []
    r = rng
    for _ in range(n):
        r, sub = jax.random.split(r)
        acts = agent.sample_actions(obs, seed=sub)[0, 0]  # (N, A)
        a_list.append(np.array(acts).reshape(-1))
    return np.asarray(a_list)


def w2_proxy(joint: np.ndarray, fact: np.ndarray) -> float:
    """Upper-bound proxy using matched seeds: sqrt(E||X-Y||^2), any dimension."""
    assert joint.shape == fact.shape
    X = joint.reshape(joint.shape[0], -1)
    Y = fact.reshape(fact.shape[0], -1)
    return float(np.sqrt(np.mean(np.sum((X - Y) ** 2, axis=1))))


def sliced_w2(joint: np.ndarray, fact: np.ndarray, n_proj: int = 64, seed: int = 0) -> float:
    """Sliced W2 with random 1D projections for arbitrary D."""
    rng = np.random.default_rng(seed)
    X = joint.reshape(joint.shape[0], -1).astype(np.float64)
    Y = fact.reshape(fact.shape[0], -1).astype(np.float64)
    assert X.shape == Y.shape
    D = X.shape[1]
    acc = 0.0
    for _ in range(n_proj):
        u = rng.normal(size=(D,))
        u = u / (np.linalg.norm(u) + 1e-8)
        x1 = X @ u
        y1 = Y @ u
        x1.sort(); y1.sort()
        acc += np.mean((x1 - y1) ** 2)
    return float(np.sqrt(acc / n_proj))


def estimate_perf(q_env_jax, samples: np.ndarray) -> float:
    vals = q_env_jax(jnp.asarray(samples))
    return float(jnp.mean(vals))


def _corr_from_samples(samples: np.ndarray) -> float:
    a1, a2 = _extract_a1_a2_from_policy_samples(samples)
    if a1.size < 2:
        return float('nan')
    return float(np.corrcoef(a1, a2)[0, 1])


def _mi_hist_from_samples(samples: np.ndarray, bins: int = 40) -> float:
    """Histogram-based MI estimate between a1 and a2 in [-1,1]."""
    a1, a2 = _extract_a1_a2_from_policy_samples(samples)
    if a1.size < 2:
        return float('nan')
    a1 = np.clip(a1, -1.0, 1.0)
    a2 = np.clip(a2, -1.0, 1.0)
    H, xedges, yedges = np.histogram2d(a1, a2, bins=bins, range=[[-1, 1], [-1, 1]], density=False)
    N = float(np.sum(H))
    if N <= 0:
        return float('nan')
    pxy = H / N
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    eps = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        mi_mat = pxy * (np.log(pxy + eps) - np.log(px @ py + eps))
    mi = np.nansum(mi_mat)
    return float(mi)

def _mi_hist_2vars(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """Histogram-based MI between two 1D variables in [-1,1]. Returns nats."""
    if a.size < 2 or b.size < 2:
        return float('nan')
    a = np.clip(np.asarray(a), -1.0, 1.0)
    b = np.clip(np.asarray(b), -1.0, 1.0)
    H, xedges, yedges = np.histogram2d(a, b, bins=bins, range=[[-1, 1], [-1, 1]], density=False)
    N = float(np.sum(H))
    if N <= 0:
        return float('nan')
    pxy = H / N
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    eps = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        mi_mat = pxy * (np.log(pxy + eps) - np.log(px @ py + eps))
    mi = np.nansum(mi_mat)
    return float(mi)

def _mi_hist_pairwise_mean(samples: np.ndarray, n_agents: int, action_dim: int, bins: int = 40) -> float:
    """Average pairwise MI over agents' first action component.

    Extracts per-agent scalar actions using the first component of each agent's
    action vector, then averages MI across all i<j pairs.

    Supports sample shapes: (S, n_agents*action_dim), (S, n_agents, action_dim), or (S, n_agents).
    """
    X = np.asarray(samples)
    # Normalize to (S, n_agents, action_dim)
    if X.ndim == 2 and X.shape[1] == n_agents * action_dim:
        Xv = X.reshape(X.shape[0], n_agents, action_dim)
    elif X.ndim == 3 and X.shape[1] == n_agents and X.shape[2] == action_dim:
        Xv = X
    elif X.ndim == 2 and X.shape[1] == n_agents:
        # Treat as action_dim=1
        Xv = X[:, :, None]
    else:
        # Fallback: try to infer n_agents from trailing dimension
        D = X.shape[-1]
        if D % action_dim == 0 and (D // action_dim) == n_agents:
            Xv = X.reshape(X.shape[0], n_agents, action_dim)
        else:
            # As a last resort, take first two agents via existing helper
            return _mi_hist_from_samples(X, bins=bins)

    S = Xv.shape[0]
    if S < 2:
        return float('nan')
    # First component per agent => (S, n_agents)
    Va = Xv[:, :, 0]
    # Compute average over all i<j pairs
    vals = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            vals.append(_mi_hist_2vars(Va[:, i], Va[:, j], bins=bins))
    if not vals:
        return float('nan')
    return float(np.nanmean(vals))


def q_env_xor_factory(coupling_c: float):
    """Return a JAX-friendly XOR reward function scaled by c.

    Reward 1 if signs differ else 0, times c.
    """
    def q_env_xor(a: jnp.ndarray) -> jnp.ndarray:
        # a: (..., 2)
        a1 = a[..., 0]
        a2 = a[..., 1]
        # Indicator of sign difference; treat 0 as non-negative for stability
        s1 = jnp.sign(a1)
        s2 = jnp.sign(a2)
        diff = jnp.not_equal(s1, s2).astype(a1.dtype)
        return coupling_c * diff
    return q_env_xor


def compute_q_grid_env(q_env_jax, grid_size: int = 129) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    A1, A2 = np.meshgrid(xs, xs, indexing='ij')
    pts = np.stack([A1.reshape(-1), A2.reshape(-1)], axis=-1)
    Q = np.array(q_env_jax(jnp.asarray(pts))).reshape(grid_size, grid_size)
    return xs, xs, Q


# ============ Agent creation and training ============ #

def create_agent(agent_name, dummy_batch, seed=0):
    # Prepare configs per agent type
    if agent_name == 'macflow':
        cfg = get_fact_cfg()
        cfg["encoder"] = None
        cfg["alpha"] = 3.0
        cfg["flow_steps"] = 10
        cfg["discount"] = 0.995
        cfg["normalize_q_loss"] = True
        cfg["q_agg"] = "mean"
        cfg["actor_hidden_dims"] = (512, 512, 512, 512)
        cfg["value_hidden_dims"] = (512, 512, 512, 512)
    elif agent_name == 'diffusion_bc':
        cfg = get_diff_cfg()
        cfg["encoder"] = None
        cfg["diffusion_steps"] = 1000
        cfg["actor_hidden_dims"] = (512, 512, 512, 512)
    else:
        raise ValueError(f'Unknown agent_name: {agent_name}')

    ex_obs = dummy_batch["observations"]
    ex_act = dummy_batch["actions"]
    # Use only a single exemplar (B=1, T=1) to initialize networks to avoid OOM
    try:
        ex_obs = ex_obs[:1, :1]
        ex_act = ex_act[:1, :1]
    except Exception:
        pass
    if ex_act.ndim == 3:
        ex_act = ex_act[..., None]
    n_agents = int(ex_act.shape[-2])
    agent_names = tuple([f"agent_{i+1}" for i in range(n_agents)])
    if agent_name == "macflow":
        agent = MACFlowAgent.create(
            seed=seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,
        )
        return agent
    elif agent_name == 'diffusion_bc':
        assert DiffusionBCAgent is not None, 'DiffusionBCAgent not available'
        agent = DiffusionBCAgent.create(
            seed=seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            agent_names=agent_names,
            config=cfg,
        )
        return agent



def train(agent: MACFlowAgent, dataset, steps: int, batch_size: int, logger: CsvLogger,
          toy: str, log_interval: int = 200, w2_samples: int = 2000, grid_size: int = 81, seed: int = 0,
          fig_dir: str = None, coupling_c: float = 1.0,
          lm_reward_c: float = 1.0, lm_collision_lambda: float = 0.0, lm_collision_radius: float = 0.15,
          lm_speed_max: float = 0.02, lm_perf_rollouts: int = 10):
    # Choose env Q and its Lipschitz estimate
    if toy == 'quadratic':
        q_env = lambda a: coupling_c * q_env_quadratic(a)
    elif toy == 'ring':
        q_env = lambda a: coupling_c * q_env_ring(a)
    elif toy == 'xor':
        q_env = q_env_xor_factory(coupling_c)
    elif toy == 'landmark':
        q_env = None  # skip env-based perf metrics for high-D toy in this script
    else:
        raise ValueError(f"Unknown toy: {toy}")

    # Precompute constant env Lipschitz once (grid-based). For XOR (non-smooth), set to 0.0.
    if toy == 'xor':
        env_L = 0.0
    elif toy in ('quadratic', 'ring'):
        env_L = estimate_env_lipschitz(q_env, grid_size=grid_size)
    else:
        env_L = 0.0
    data_size = dataset['actions'].shape[0]
    rng = jax.random.PRNGKey(seed)

    history = _init_history()

    for step in range(steps):
        idx = np.random.randint(0, data_size, size=batch_size)
        batch = {k: v[idx] for k, v in dataset.items()}
        if step == 0:
            try:
                bshape = batch['actions'].shape
                print(f"[debug] initial batch shape actions={bshape}, requested batch_size={batch_size}")
            except Exception:
                pass
        agent, info = agent.update(batch, step)

        if (step % log_interval == 0) or (step == steps - 1):
            # Samples for metrics
            rng, sub1, sub2 = jax.random.split(rng, 3)
            obs_template = dataset['observations'][:1, :1]
            has_joint = hasattr(agent, 'compute_flow_actions')
            # Some agents (e.g., diffusion_bc) have no critic; guard usages
            params = getattr(agent.network, 'params', {}) if hasattr(agent, 'network') else {}
            try:
                has_critic = ('modules_critic' in params)
            except Exception:
                has_critic = False
            if has_joint:
                joint_samps = sample_joint_actions(agent, n=w2_samples, rng=sub1, obs_template=obs_template)
            else:
                joint_samps = None
            fact_samps = sample_factored_actions(agent, n=w2_samples, rng=sub2, obs_template=obs_template)
            # Save figures per toy
            if toy == 'xor' and fig_dir is not None and has_joint and (joint_samps is not None):
                _save_xor_samples_figure(dataset['actions'], joint_samps, fact_samps, fig_dir)
                # Save raw samples for quick inspection
                try:
                    np.save(os.path.join(fig_dir, 'joint_samples.npy'), np.asarray(joint_samps))
                    np.save(os.path.join(fig_dir, 'factored_samples.npy'), np.asarray(fact_samps))
                except Exception:
                    pass
                # Also save Q heatmap: learned critic if available, else env Q
                _save_xor_q_heatmap(agent, fig_dir, obs_template=obs_template, grid_size=grid_size,
                                    use_env=(not has_critic), q_env_jax=q_env if (not has_critic) else None)
            elif (
                fig_dir is not None
                and has_joint
                and (joint_samps is not None)
                and joint_samps.ndim == 2
                and joint_samps.shape[1] == 2
                and toy in ('quadratic', 'ring')
            ):
                _save_policy_distribution(joint_samps, fig_dir, name='joint', toy=toy)
                _save_policy_distribution(fact_samps, fig_dir, name='factored', toy=toy)
                _save_policy_scatter_density(joint_samps, fig_dir, name='joint', toy=toy)
                _save_policy_scatter_density(fact_samps, fig_dir, name='factored', toy=toy)

            # Metrics
            if has_joint:
                w2_p = w2_proxy(joint_samps, fact_samps)
                sw2 = sliced_w2(joint_samps, fact_samps, n_proj=64, seed=seed)
            else:
                w2_p = float('nan')
                sw2 = float('nan')
            if q_env is not None and has_joint:
                perf_joint = estimate_perf(q_env, joint_samps)
                perf_fact = estimate_perf(q_env, fact_samps)
                perf_gap = perf_joint - perf_fact
            elif toy == 'landmark' and hasattr(agent, 'network') and hasattr(agent.network, 'params') and has_joint:
                # Estimate performance via environment rollouts and reward
                perf_joint = _estimate_landmark_perf(agent, dataset, mode='joint',
                                                     n_rollouts=min(lm_perf_rollouts, data_size),
                                                     lm_speed_max=lm_speed_max,
                                                     lm_reward_c=lm_reward_c,
                                                     lm_collision_lambda=lm_collision_lambda,
                                                     lm_collision_radius=lm_collision_radius)
                perf_fact = _estimate_landmark_perf(agent, dataset, mode='factored',
                                                     n_rollouts=min(lm_perf_rollouts, data_size),
                                                     lm_speed_max=lm_speed_max,
                                                     lm_reward_c=lm_reward_c,
                                                     lm_collision_lambda=lm_collision_lambda,
                                                     lm_collision_radius=lm_collision_radius)
                perf_gap = perf_joint - perf_fact
            else:
                # For agents without joint policy, compute only factored perf
                if q_env is not None:
                    perf_joint = float('nan')
                    perf_fact = estimate_perf(q_env, fact_samps)
                    perf_gap = float('nan')
                elif toy == 'landmark':
                    perf_joint = float('nan')
                    perf_fact = _estimate_landmark_perf(agent, dataset, mode='factored',
                                                        n_rollouts=min(lm_perf_rollouts, data_size),
                                                        lm_speed_max=lm_speed_max,
                                                        lm_reward_c=lm_reward_c,
                                                        lm_collision_lambda=lm_collision_lambda,
                                                        lm_collision_radius=lm_collision_radius)
                    perf_gap = float('nan')
                else:
                    perf_joint = float('nan'); perf_fact = float('nan'); perf_gap = float('nan')
            if has_joint and has_critic and toy in ('quadratic', 'ring', 'xor') and (joint_samps is not None and joint_samps.ndim == 2 and joint_samps.shape[1] == 2):
                crit_L = estimate_critic_lipschitz(agent, grid_size=grid_size)
            elif has_joint and has_critic and toy == 'landmark':
                crit_L = estimate_critic_lipschitz_nd(agent, obs_template=obs_template, num_points=128, seed=seed)
            else:
                crit_L = 0.0
            # Dependency metrics
            corr_joint = _corr_from_samples(joint_samps) if has_joint else float('nan')
            corr_fact = _corr_from_samples(fact_samps)
            # Pairwise MI averaged over agent pairs using first action component
            N = int(agent.config['num_agents'])
            A = int(agent.config['action_dim'])
            mi_joint = _mi_hist_pairwise_mean(joint_samps, n_agents=N, action_dim=A, bins=40) if has_joint else float('nan')
            mi_fact = _mi_hist_pairwise_mean(fact_samps, n_agents=N, action_dim=A, bins=40)

            # Debug: capture current (B,T) and sample counts
            try:
                cur_B = int(batch['actions'].shape[0])
                cur_T = int(batch['actions'].shape[1])
            except Exception:
                cur_B, cur_T = -1, -1

            log_row = {k: float(v) for k, v in info.items()}
            log_row.update({
                'metric/env_lipschitz': env_L,
                'metric/critic_lipschitz': crit_L,
                'metric/w2_proxy': w2_p,
                'metric/sliced_w2': sw2,
                'metric/perf_joint': perf_joint,
                'metric/perf_fact': perf_fact,
                'metric/perf_gap': perf_gap,
                'metric/corr_joint': corr_joint,
                'metric/corr_fact': corr_fact,
                'metric/mi_joint': mi_joint,
                'metric/mi_fact': mi_fact,
                'debug/batch_B': float(cur_B),
                'debug/batch_T': float(cur_T),
                'debug/w2_n': float(0 if (joint_samps is None) else np.asarray(joint_samps).shape[0]),
            })
            logger.log(log_row, step)
            # Update and save curves
            _update_history(history, step, log_row)
            if fig_dir is not None:
                if toy != 'xor':
                    _save_curves(history, fig_dir, title_suffix=f' [{toy}]')
                if toy == 'landmark':
                    _save_theory_curves(history, fig_dir, title_suffix=' [landmark]')
            print(
                f"[step {step:05d}] L_env={env_L:.3f} L_crit={crit_L:.3f} "
                f"W2≈{w2_p:.3f} SW2≈{sw2:.3f} gap={perf_gap:.4f}"
            )
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--toy', type=str, choices=['quadratic', 'ring', 'xor', 'landmark'], required=True)
    p.add_argument('--agent', type=str, choices=['macflow', 'diffusion_bc'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_dataset', type=int, default=50000)
    p.add_argument('--save_dir', type=str, default='exp/toy_metrics')
    p.add_argument('--grid_size', type=int, default=81)
    p.add_argument('--w2_samples', type=int, default=2000)
    p.add_argument('--log_interval', type=int, default=200)
    # Quadratic dataset knob
    p.add_argument('--corr_sigma', type=float, default=0.15)
    # Ring dataset knob
    p.add_argument('--ring_noise', type=float, default=0.03)
    # XOR dataset knobs
    p.add_argument('--coupling_c', type=float, default=1.0)
    p.add_argument('--noise_p', type=float, default=0.0)
    p.add_argument('--imbalance_q', type=float, default=0.5)
    # Landmark dataset knobs
    p.add_argument('--lm_T', type=int, default=50)
    p.add_argument('--lm_noise_sigma', type=float, default=0.02)
    p.add_argument('--lm_swap_q', type=float, default=0.0)
    p.add_argument('--lm_reward_c', type=float, default=1.0)
    p.add_argument('--lm_collision_lambda', type=float, default=0.0)
    p.add_argument('--lm_collision_radius', type=float, default=0.15)
    p.add_argument('--lm_speed_max', type=float, default=0.05)
    p.add_argument('--start_sigma', type=float, default=0.03)
    p.add_argument('--lm_n_agents', type=int, default=3)
    p.add_argument('--lm_start_mode', type=str, default='center',
                   choices=['auto', 'center', 'quad_centers', 'layout_centers'],
                   help='Start placement: auto | center(0,0) | quad_centers(±sep,±sep) | layout_centers')
    # Landmark grid / layout
    p.add_argument('--grid', action='store_true')
    p.add_argument('--grid_noise_sigmas', type=str, default='')
    p.add_argument('--grid_swap_qs', type=str, default='')
    p.add_argument('--grid_seeds', type=str, default='')
    p.add_argument('--lm_num_landmarks', type=int, default=0, help='If >0, overrides number of landmarks (M)')
    p.add_argument('--lm_layout', type=str, default='auto', choices=['auto', 'ring', 'double_ring', 'triple_ring', 'twin_double_ring', 'quad_double_ring', 'triangle'])
    p.add_argument('--lm_radii', type=str, default='', help='Comma-separated radii for rings e.g., 0.6,1.2')
    p.add_argument('--lm_counts', type=str, default='', help='Comma-separated counts per ring e.g., 8,12')
    p.add_argument('--lm_include_center', action='store_true')
    p.add_argument('--lm_twin_center_sep', type=float, default=0.8, help='Half-distance between the two double-ring centers in twin_double_ring (default centers at x=±0.8)')
    p.add_argument('--lm_twin_min_gap', type=float, default=0.10, help='Min gap so that r_out <= sep - gap to avoid overlap between twin rings')
    p.add_argument('--lm_ring_band_min', type=float, default=0.20, help='Min thickness between inner and outer rings: r_out - r_in >= band')
    p.add_argument('--lm_quad_center_sep', type=float, default=0.75, help='Distance from origin to each quadrant center in quad_double_ring (default 0.75)')
    p.add_argument('--lm_quad_min_gap', type=float, default=0.10, help='Min gap for quad double rings so r_out <= sep - gap')
    p.add_argument('--lm_reward_mode', type=str, default='auto', choices=['auto', 'optimal', 'greedy', 'fixed', 'nearest'])
    # Landmark plot mode
    p.add_argument('--lm_plot_mode', type=str, default='overlay', choices=['overlay', 'panels', 'both'],
                   help='Landmark plotting: overlay (single axes), panels (clustered subplots), or both')
    # Plot control: subsample trajectories to avoid heavy figures
    p.add_argument('--plot_max_traj_dataset', type=int, default=10,
                   help='If >0, limits trajectories per agent for dataset plots')
    p.add_argument('--plot_max_traj_policy', type=int, default=10,
                   help='If >0, limits trajectories per agent for policy plots')
    p.add_argument('--plot_t_stride', type=int, default=1,
                   help='Plot every t_stride time steps in trajectory plots')
    args = p.parse_args()

    def run_one(a):
        os.makedirs(a.save_dir, exist_ok=True)
        exp_dir = os.path.join(a.save_dir, f"{a.toy}_{get_exp_name(a.seed)}")
        os.makedirs(exp_dir, exist_ok=True)
        csv_path = os.path.join(exp_dir, 'metrics.csv')
        logger = CsvLogger(csv_path)

        print(f"Building dataset: {a.toy}")
        if a.toy == 'quadratic':
            dataset = make_dataset_quadratic(n_samples=a.n_dataset, T=2, corr_sigma=a.corr_sigma, seed=a.seed,
                                             noise_p=a.noise_p, imbalance_q=a.imbalance_q, coupling_c=a.coupling_c)
        elif a.toy == 'ring':
            dataset = make_dataset_ring(n_samples=a.n_dataset, T=2, noise_sigma=a.ring_noise, seed=a.seed,
                                        noise_p=a.noise_p, imbalance_q=a.imbalance_q, coupling_c=a.coupling_c)
        elif a.toy == 'xor':
            dataset = make_dataset_xor(n_samples=a.n_dataset, T=2,
                                       coupling_c=a.coupling_c,
                                       noise_p=a.noise_p,
                                       imbalance_q=a.imbalance_q,
                                       seed=a.seed)
        else:  # landmark
            # For fair compute, default to a smaller dataset size for landmark unless user overrides
            # Parse optional radii/counts
            lm_radii = [float(x) for x in a.lm_radii.split(',') if x.strip() != ''] if a.lm_radii else None
            lm_counts = [int(x) for x in a.lm_counts.split(',') if x.strip() != ''] if a.lm_counts else None
            dataset = make_dataset_landmark(n_episodes=int(a.n_dataset), T=a.lm_T, seed=a.seed,
                                            start_sigma=a.start_sigma,
                                            start_mode=a.lm_start_mode,
                                            lm_noise_sigma=a.lm_noise_sigma, lm_swap_q=a.lm_swap_q,
                                            lm_reward_c=a.lm_reward_c,
                                            lm_collision_lambda=a.lm_collision_lambda,
                                            lm_collision_radius=a.lm_collision_radius,
                                            lm_speed_max=a.lm_speed_max,
                                            n_agents=a.lm_n_agents,
                                            lm_num_landmarks=(a.lm_num_landmarks if a.lm_num_landmarks > 0 else None),
                                            lm_layout=a.lm_layout,
                                            lm_radii=lm_radii,
                                            lm_counts=lm_counts,
                                            lm_twin_center_sep=a.lm_twin_center_sep,
                                            lm_twin_min_gap=a.lm_twin_min_gap,
                                            lm_ring_band_min=a.lm_ring_band_min,
                                            lm_quad_center_sep=a.lm_quad_center_sep,
                                            lm_quad_min_gap=a.lm_quad_min_gap,
                                            lm_include_center=a.lm_include_center,
                                            reward_mode=a.lm_reward_mode)

        # Save dataset distribution figures once (2D toys). Landmark preview is separate.
        if a.toy in ('quadratic', 'ring', 'xor'):
            _save_dataset_distribution(dataset, exp_dir, toy=a.toy)
        elif a.toy == 'landmark':
            _save_landmark_preview(dataset, exp_dir)
            if a.lm_plot_mode in ('overlay', 'both'):
                _save_landmark_trajectories_dataset(
                    dataset, exp_dir,
                    n_show=min(50, a.n_dataset),
                    style='scatter',
                    max_trajectories=(a.plot_max_traj_dataset if a.plot_max_traj_dataset > 0 else None),
                    t_stride=max(1, a.plot_t_stride),
                )
                # Save line-only version as well
                _save_landmark_trajectories_dataset(
                    dataset, exp_dir,
                    n_show=min(50, a.n_dataset),
                    style='line',
                    max_trajectories=(a.plot_max_traj_dataset if a.plot_max_traj_dataset > 0 else None),
                    t_stride=max(1, a.plot_t_stride),
                )
            if a.lm_plot_mode in ('panels', 'both'):
                _save_landmark_trajectories_dataset_panels(
                    dataset, exp_dir,
                    n_show=min(50, a.n_dataset),
                    max_trajectories=(a.plot_max_traj_dataset if a.plot_max_traj_dataset > 0 else None),
                    t_stride=max(1, a.plot_t_stride),
                )

        print("Creating joint MACFlow agent...")
        agent = create_agent(a.agent, dataset, seed=a.seed)

        print("Training and logging metrics...")
        t0 = time.time()
        agent = train(agent, dataset, steps=a.steps, batch_size=a.batch_size, logger=logger,
                  toy=a.toy, log_interval=a.log_interval, w2_samples=a.w2_samples,
                  grid_size=a.grid_size, seed=a.seed, fig_dir=exp_dir,
                  coupling_c=a.coupling_c,
                  lm_reward_c=a.lm_reward_c,
                  lm_collision_lambda=a.lm_collision_lambda,
                  lm_collision_radius=a.lm_collision_radius,
                  lm_speed_max=a.lm_speed_max,
                  lm_perf_rollouts=10)
        if a.toy == 'landmark':
            has_joint = hasattr(agent, 'compute_flow_actions')
            if a.lm_plot_mode in ('overlay', 'both'):
                if has_joint:
                    _save_landmark_trajectories_policy(
                        agent, dataset, exp_dir, mode='joint', n_rollouts=30, lm_speed_max=a.lm_speed_max,
                        style='scatter',
                        max_trajectories=(a.plot_max_traj_policy if a.plot_max_traj_policy > 0 else None),
                        t_stride=max(1, a.plot_t_stride),
                    )
                _save_landmark_trajectories_policy(
                    agent, dataset, exp_dir, mode='factored', n_rollouts=30, lm_speed_max=a.lm_speed_max,
                    style='scatter',
                    max_trajectories=(a.plot_max_traj_policy if a.plot_max_traj_policy > 0 else None),
                    t_stride=max(1, a.plot_t_stride),
                )
                # Save line-only variants
                if has_joint:
                    _save_landmark_trajectories_policy(
                        agent, dataset, exp_dir, mode='joint', n_rollouts=30, lm_speed_max=a.lm_speed_max,
                        style='line',
                        max_trajectories=(a.plot_max_traj_policy if a.plot_max_traj_policy > 0 else None),
                        t_stride=max(1, a.plot_t_stride),
                    )
                _save_landmark_trajectories_policy(
                    agent, dataset, exp_dir, mode='factored', n_rollouts=30, lm_speed_max=a.lm_speed_max,
                    style='line',
                    max_trajectories=(a.plot_max_traj_policy if a.plot_max_traj_policy > 0 else None),
                    t_stride=max(1, a.plot_t_stride),
                )
            if a.lm_plot_mode in ('panels', 'both'):
                if has_joint:
                    _save_landmark_trajectories_policy_panels(
                        agent, dataset, exp_dir, mode='joint', n_rollouts=30, lm_speed_max=a.lm_speed_max,
                        max_trajectories=(a.plot_max_traj_policy if a.plot_max_traj_policy > 0 else None),
                        t_stride=max(1, a.plot_t_stride),
                    )
                _save_landmark_trajectories_policy_panels(
                    agent, dataset, exp_dir, mode='factored', n_rollouts=30, lm_speed_max=a.lm_speed_max,
                    max_trajectories=(a.plot_max_traj_policy if a.plot_max_traj_policy > 0 else None),
                    t_stride=max(1, a.plot_t_stride),
                )
        print(f"Done in {time.time() - t0:.1f}s. Logs at {csv_path}")

    # Grid mode for landmark toy
    if args.toy == 'landmark' and args.grid:
        noise_vals = [float(x) for x in args.grid_noise_sigmas.split(',') if x.strip() != ''] if args.grid_noise_sigmas else [args.lm_noise_sigma]
        swap_vals = [float(x) for x in args.grid_swap_qs.split(',') if x.strip() != ''] if args.grid_swap_qs else [args.lm_swap_q]
        seed_vals = [int(x) for x in args.grid_seeds.split(',') if x.strip() != ''] if args.grid_seeds else [args.seed]
        for nv in noise_vals:
            for qv in swap_vals:
                for sv in seed_vals:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.seed = sv
                    run_args.lm_noise_sigma = nv
                    run_args.lm_swap_q = qv
                    # disambiguate save dir
                    tag = f"ns{nv}_q{qv}_s{sv}"
                    run_args.save_dir = os.path.join(args.save_dir, tag)
                    run_one(run_args)
    else:
        run_one(args)


if __name__ == '__main__':
    main()

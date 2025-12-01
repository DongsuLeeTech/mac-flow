from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution

class DiscreteActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2
    use_relaxed: bool = False
    encoder: Optional[nn.Module] = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logits_net = nn.Dense(self.action_dim,
                                   kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, temperature: float = 1.0):
        x = self.encoder(observations) if self.encoder else observations
        hidden = self.actor_net(x)
        logits = self.logits_net(hidden) / temperature

        if self.use_relaxed:
            distribution = distrax.RelaxedOneHotCategorical(
                temperature=temperature,
                logits=logits
            )
        else:
            distribution = distrax.Categorical(logits=logits)

        return distribution

class SoftmaxActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: Optional[nn.Module] = None

    def setup(self):
        mlp_class = MLP
        policy_net = mlp_class((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        self.policy_net = policy_net

    def __call__(self, observations):
        x = self.encoder(observations) if self.encoder else observations
        logits = self.policy_net(x)
        probs = nn.softmax(logits, axis=-1)
        return probs

class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v

class ActorVectorField(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v

class DQN(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        return v

class QMixer(nn.Module):
    state_dim: int
    n_agents: int
    embed_dim: int
    hypernet_embed: int
    layer_norm: bool = True
    non_monotonic: bool = False

    def setup(self):
        mlp_class = MLP
        mixer_ = mlp_class((self.hypernet_embed,
                            self.embed_dim * self.n_agents),
                            activate_final=False, layer_norm=self.layer_norm)
        mixer_final = mlp_class((self.hypernet_embed, self.embed_dim), activate_final=False, layer_norm=self.layer_norm)
        b_ = nn.Dense(self.embed_dim, kernel_init=default_init())
        V_ = mlp_class((self.embed_dim, 1), activate_final=False, layer_norm=self.layer_norm)
        self.mixer_ = mixer_
        self.mixer_final = mixer_final
        self.b_ = b_
        self.V_ = V_

    def __call__(self, states, agent_qs=None):
        if agent_qs is not None:
            return self.b(agent_qs, states)
        else:
            return self.k(states)

    def b(self, agent_qs, states):
        B = agent_qs.shape[0]
        state_dim = states.shape[2:]
        agent_qs = jnp.reshape(agent_qs, (-1, 1, self.n_agents))
        states = jnp.reshape(states, (-1, *state_dim))

        w1 = self.mixer_(states)
        if not self.non_monotonic:
            w1 = jnp.abs(w1)
        b1 = self.b_(states)
        w1 = jnp.reshape(w1, (-1, self.n_agents, self.embed_dim))
        b1 = jnp.reshape(b1, (-1, 1, self.embed_dim))
        hidden = nn.elu(jnp.matmul(agent_qs, w1) + b1)

        w_final = self.mixer_final(states)
        if not self.non_monotonic:
            w_final = jnp.abs(w_final)
        w_final = jnp.reshape(w_final, (-1, self.embed_dim, 1))

        v = jnp.reshape(self.V_(states), (-1, 1, 1))
        y = jnp.matmul(hidden, w_final) + v
        q_tot = jnp.reshape(y, (B, -1, 1))
        return q_tot

    def k(self, states):
        B, T = states.shape[:2]

        w1 = jnp.abs(self.mixer_(states))
        w_final = jnp.abs(self.mixer_final(states))
        w1 = jnp.reshape(w1, (-1, self.n_agents, self.embed_dim))
        w_final = jnp.reshape(w_final, (-1, self.embed_dim, 1))
        k = jnp.matmul(w1, w_final)
        k = jnp.reshape(k, (B, -1, self.n_agents))
        k = k / (jnp.sum(k, axis=2, keepdims=True) + 1e-10)
        return k

class MixNet(nn.Module):
    state_dim: int
    hidden_dims: int
    num_agents: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        self.f_v = mlp_class((self.state_dim, self.hidden_dims), activate_final=False,
                             layer_norm=self.layer_norm)
        self.w_v = mlp_class((self.hidden_dims, self.num_agents), activate_final=False,
                             layer_norm=self.layer_norm)
        self.b_v =mlp_class((self.hidden_dims, 1), activate_final=False,
                             layer_norm=self.layer_norm)

    def __call__(self, states):
        x = self.f_v(states)
        w = self.w_v(x)
        b = self.b_v(x)
        return jnp.abs(w), b



class SequenceLSTMEncoder(nn.Module):
    """Sequence encoder with optional per-step encoder and stacked LSTM.

    Inputs are expected to be time-major: (T, B, N, ...), but any leading batch
    shape is supported as long as time is the first dimension.

    - Optionally applies a `point_encoder` to each timestep independently
      (e.g., CNN/MLP) before the recurrent layers.
    - Stacked LSTM(s) then process features over time with reset masking.
    - Returns per-timestep hidden states of the top LSTM layer with the same
      leading shape as inputs, replacing the feature dimension.
    """

    hidden_dim: int
    num_layers: int = 1
    pre_mlp_dims: Sequence[int] = ()
    layer_norm: bool = False
    point_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, resets: Optional[jnp.ndarray] = None,
                 initial_carry: Optional[Sequence[Any]] = None,
                 return_carry: bool = False):
        # observations: (T, B, N, F...) -> flatten trailing feature dims if needed
        # Accept arbitrary leading dims after T; combine them to an effective batch M
        x = observations
        # Flatten any spatial dims before MLP/LSTM; apply point encoder first if given
        if self.point_encoder is not None:
            # point_encoder should broadcast over leading dims automatically
            x = self.point_encoder(x)
        # Ensure last dim is feature
        x = x.reshape(x.shape[: -1] + (-1,)) if x.ndim >= 2 else x

        T = x.shape[0]
        leading_shape = x.shape[1:-1]  # e.g., (B, N)
        # compute M using Python ints from shape, avoiding JAX tracers
        M = 1
        if len(leading_shape) > 0:
            for d in leading_shape:
                M *= d
        feat_dim = x.shape[-1]

        x_flat = x.reshape((T, M, feat_dim))

        if self.pre_mlp_dims:
            x_flat = MLP(self.pre_mlp_dims, activate_final=True, layer_norm=self.layer_norm)(x_flat)

        # Simple, version-agnostic zero carry with batch dim M
        def _zero_carry(batch_size: int, hidden: int, dtype):
            c = jnp.zeros((batch_size, hidden), dtype)
            h = jnp.zeros((batch_size, hidden), dtype)
            return (c, h)

        # Wrapper cell that applies reset mask before calling LSTMCell
        class _ResetLSTMCell(nn.Module):
            hidden_dim: int
            @nn.compact
            def __call__(self, carry, inputs):
                x_t, reset_t = inputs  # (M, D), (M,)
                lstm = nn.OptimizedLSTMCell(self.hidden_dim)

                # Support both tuple and object state
                def _get_c_h(s):
                    if hasattr(s, 'c') and hasattr(s, 'h'):
                        return s.c, s.h
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        return s[0], s[1]
                    raise TypeError(f"Unrecognized LSTM state structure: {type(s)}")

                def _make_state_like(s, c, h):
                    if hasattr(s, 'c') and hasattr(s, 'h'):
                        cls = type(s)
                        try:
                            return cls(c=c, h=h)
                        except TypeError:
                            try:
                                return cls(c, h)
                            except TypeError:
                                return (c, h)
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        return (c, h)
                    return (c, h)

                c, h = _get_c_h(carry)
                c = jnp.where(reset_t[:, None], jnp.zeros_like(c), c)
                h = jnp.where(reset_t[:, None], jnp.zeros_like(h), h)
                carry = _make_state_like(carry, c, h)

                new_carry, y = lstm(carry, x_t)
                return new_carry, y

        if resets is None:
            resets_flat = jnp.zeros((T, M), dtype=bool)
            if initial_carry is None:
                resets_flat = resets_flat.at[0, :].set(True)
        else:
            assert resets.shape[: len(leading_shape) + 1] == (T, *leading_shape), (
                f"resets shape should be (T, *leading_shape), got {resets.shape}, expected {(T, *leading_shape)}"
            )
            resets_flat = resets.reshape((T, M)).astype(bool)
            if initial_carry is None:
                resets_flat = resets_flat.at[0, :].set(True)

        # Prepare initial carries
        if initial_carry is None:
            carries = tuple(_zero_carry(M, self.hidden_dim, x_flat.dtype) for _ in range(self.num_layers))
        else:
            carries = tuple(initial_carry)

        # Apply stacked LSTMs with nn.scan over time
        y = x_flat
        new_carries = []
        for i in range(self.num_layers):
            ScannedCell = nn.scan(
                _ResetLSTMCell,
                variable_broadcast=('params',),
                split_rngs={'params': False},
                in_axes=0,
                out_axes=0,
            )
            scan_cell = ScannedCell(self.hidden_dim, name=f'lstm_scan_{i}')
            carry_i, y = scan_cell(carries[i], (y, resets_flat))
            new_carries.append(carry_i)

        out = y.reshape((T, *leading_shape, self.hidden_dim))
        final_carry = tuple(new_carries)
        if return_carry:
            return out, final_carry
        return out

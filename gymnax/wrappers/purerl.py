"""Jittable wrappers that keep the functional Gymnax API."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from gymnax.environments import environment, spaces


class GymnaxWrapper:
    """Base class for Gymnax wrappers.

    Attributes and methods that a wrapper does not override are forwarded to the
    wrapped environment.

    Args:
        env (Environment): Environment to wrap.
    """

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class StickyActionState:
    """State of an environment wrapped in `StickyActionWrapper`.

    Attributes:
        env_state: State of the wrapped environment.
        last_action: Action executed in the previous step, 0 after a reset.
    """

    env_state: Any
    last_action: jax.Array


class StickyActionWrapper(GymnaxWrapper):
    """Repeat the previously executed action with a fixed probability.

    At every step the requested action is replaced by the action executed in the
    previous step with probability `sticky_action_prob`. Requires a `Discrete`
    action space.

    Args:
        env (Environment): Environment to wrap.
        sticky_action_prob: Probability in `[0, 1]` of repeating the previous
            action.

    Raises:
        ValueError: If the probability is not a finite value in `[0, 1]` or the
            action space is not `Discrete`.
    """

    def __init__(self, env, sticky_action_prob: float = 0.0):
        super().__init__(env)
        try:
            probability = float(sticky_action_prob)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "sticky_action_prob must be a finite value in [0, 1]"
            ) from error
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("sticky_action_prob must be a finite value in [0, 1]")

        action_space = env.action_space(env.default_params)
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("StickyActionWrapper requires a Discrete action space")
        if action_space.n < 1:
            raise ValueError(
                "StickyActionWrapper requires at least one discrete action"
            )

        self._sticky_action_prob = probability
        self._action_dtype = action_space.dtype

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, StickyActionState]:
        """Reset the wrapped environment and set the previous action to 0."""
        obs, env_state = self._env.reset(key, params)
        return obs, StickyActionState(
            env_state=env_state,
            last_action=jnp.array(0, dtype=self._action_dtype),
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: StickyActionState,
        action: int | jax.Array,
        params: environment.EnvParams | None = None,
    ) -> tuple[
        jax.Array, StickyActionState, jax.Array, jax.Array, jax.Array, dict[Any, Any]
    ]:
        """Step with the requested action or, when it sticks, the previous one."""
        requested_action = jnp.asarray(action, dtype=self._action_dtype)
        if self._sticky_action_prob == 0.0:
            executed_action = requested_action
            step_key = key
        else:
            replay_key, step_key = jax.random.split(key)
            replay = jax.random.bernoulli(replay_key, self._sticky_action_prob)
            executed_action = jnp.where(replay, state.last_action, requested_action)
        obs, env_state, reward, terminated, truncated, info = self._env.step(
            step_key, state.env_state, executed_action, params
        )
        return (
            obs,
            StickyActionState(env_state=env_state, last_action=executed_action),
            reward,
            terminated,
            truncated,
            info,
        )


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten observations into one-dimensional arrays.

    Applies to the observations returned by `reset` and `step`, to
    `info["final_observation"]` and to the bounds of the observation space. Only
    `Box` observation spaces are supported.

    Args:
        env (Environment): Environment to wrap.
    """

    #   def __init__(self, env: environment.Environment):
    #     super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        """Observation space of the wrapped environment with a flat shape."""
        space = self._env.observation_space(params)
        assert isinstance(space, spaces.Box), "Only Box spaces are supported for now."

        def flatten_bound(bound):
            if jnp.ndim(bound) == 0:
                return bound
            return jnp.broadcast_to(bound, space.shape).reshape(-1)

        return spaces.Box(
            low=flatten_bound(space.low),
            high=flatten_bound(space.high),
            shape=(np.prod(space.shape),),
            dtype=space.dtype,
        )

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, environment.EnvState]:
        """Reset the wrapped environment and flatten the observation."""
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[jax.Array, environment.EnvState, float, bool, bool, Any]:  # dict]:
        """Step the wrapped environment and flatten both observations."""
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        obs = jnp.reshape(obs, (-1,))
        final_observation = jnp.reshape(info["final_observation"], (-1,))
        info = {**info, "final_observation": final_observation}
        return obs, state, reward, terminated, truncated, info


@struct.dataclass
class LogEnvState:
    """State of an environment wrapped in `LogWrapper`.

    Attributes:
        env_state: State of the wrapped environment.
        episode_returns: Sum of rewards in the running episode.
        episode_lengths: Number of steps in the running episode.
        returned_episode_returns: Return of the most recently completed episode.
        returned_episode_lengths: Length of the most recently completed episode.
        returned_episode_valid: Whether an episode has completed since the wrapper
            was reset.
    """

    env_state: environment.EnvState
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array
    returned_episode_valid: jax.Array


class LogWrapper(GymnaxWrapper):
    """Track episode returns and lengths and report them through `info`.

    `step` adds four entries to `info`:

    - `returned_episode_returns`: return of the most recently completed episode.
    - `returned_episode_lengths`: length of the most recently completed episode.
    - `returned_episode`: whether this transition completed an episode.
    - `returned_episode_valid`: whether any episode has completed so far. Until
      then the two counters above hold the placeholder value 0.

    Args:
        env (Environment): Environment to wrap.
    """

    #   def __init__(self, env: environment.Environment):
    #     super().__init__(env)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, LogEnvState]:
        """Reset the wrapped environment and all episode statistics."""
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state,
            jnp.array(0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False, dtype=jnp.bool_),
        )
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[jax.Array, LogEnvState, jax.Array, bool, bool, dict[Any, Any]]:
        """Step the wrapped environment and update the episode statistics.

        Args:
            key: PRNG key.
            state: Current wrapper state.
            action: Action to take.
            params: Environment parameters. Defaults to the environment defaults.

        Returns:
            A tuple `(obs, state, reward, terminated, truncated, info)`, where `info`
            additionally holds the episode statistics.
        """
        obs, env_state, reward, terminated, truncated, info = self._env.step(
            key, state.env_state, action, params
        )
        done = jnp.logical_or(terminated, truncated)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            returned_episode_valid=jnp.logical_or(state.returned_episode_valid, done),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        info["returned_episode_valid"] = state.returned_episode_valid
        return obs, state, reward, terminated, truncated, info

"""Wrappers for pure RL."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from gymnax.environments import environment, spaces


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class StickyActionState:
    """State for :class:`StickyActionWrapper`."""

    env_state: Any
    last_action: jax.Array


class StickyActionWrapper(GymnaxWrapper):
    """Replay the previous discrete action with a configurable probability."""

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
        """Reset the inner environment and initialize the replay action to zero."""
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
        """Step with either the requested action or the recorded previous action."""
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
    """Flatten the observations of the environment."""

    #   def __init__(self, env: environment.Environment):
    #     super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(self._env.observation_space(params), spaces.Box), (
            "Only Box spaces are supported for now."
        )
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, environment.EnvState]:
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
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        obs = jnp.reshape(obs, (-1,))
        final_observation = jnp.reshape(info["final_observation"], (-1,))
        info = {**info, "final_observation": final_observation}
        return obs, state, reward, terminated, truncated, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    #   def __init__(self, env: environment.Environment):
    #     super().__init__(env)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state,
            jnp.array(0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
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
        """Step the environment.


        Args:
          key: Pkey key.
          state: The current state of the environment.
          action: The action to take.
          params: The parameters of the environment.


        Returns:
          A tuple of (observation, state, reward, terminated, truncated, info).
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
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, terminated, truncated, info

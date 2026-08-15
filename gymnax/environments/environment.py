"""Abstract base class for all gymnax Environments."""

from functools import partial
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
from flax import struct

TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


@struct.dataclass
class EnvState:
    time: int


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 1


class Environment(Generic[TEnvState, TEnvParams]):
    """Abstract base class for environments."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> tuple[Any, TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, info = self.step_env(
            key_step, state, action, params
        )
        truncated = self.is_truncated(state_st, params)
        done = jnp.logical_or(terminated, truncated)
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree.map(
            lambda reset_leaf, step_leaf: jax.lax.select(done, reset_leaf, step_leaf),
            obs_re,
            obs_st,
        )
        info = {
            **info,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": obs_st,
        }

        return obs, state, reward, terminated, truncated, info

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: TEnvParams | None = None
    ) -> tuple[Any, TEnvState]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params

        # Reset
        obs, state = self.reset_env(key, params)

        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> tuple[Any, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[jax.Array, TEnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    def observe(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array | None,
        params: TEnvParams,
    ) -> Any:
        """Return the observation for a state and its producing action.

        Built-ins that still implement ``get_obs`` are adapted here; new
        environments should implement this keyed method directly.
        """
        del action
        return self.get_obs(state, params=params, key=key)

    def get_obs(self, state, params=None, key=None) -> Any:
        """Legacy observation hook retained for existing environment classes."""
        raise NotImplementedError

    def is_truncated(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Check whether the transition reached its configured time limit."""
        return state.time >= params.max_steps_in_episode

    def is_terminal(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Legacy natural-terminal helper; no longer includes time limits."""
        return self.is_terminated(state, params)

    def is_terminated(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Optional helper for natural terminal conditions."""
        raise NotImplementedError

    def discount(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Legacy natural-terminal discount helper for transition metadata."""
        return jax.lax.select(self.is_terminated(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params: TEnvParams):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params: TEnvParams):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: TEnvParams):
        """State space of the environment."""
        raise NotImplementedError

"""Abstract base class for all gymnax Environments."""

from functools import partial
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)

import jax
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
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: TEnvParams | None = None
    ) -> tuple[jax.Array, TEnvState]:
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
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[jax.Array, TEnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    @overload
    def get_obs(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> jax.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @overload
    def get_obs(
        self,
        state: TEnvState,
    ) -> jax.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @overload
    def get_obs(
        self, state: TEnvState, key: jax.Array, params: TEnvParams
    ) -> jax.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def get_obs(self, state, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    def discount(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

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

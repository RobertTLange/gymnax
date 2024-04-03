"""Abstract base class for all gymnax Environments."""

import functools
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
import chex
from flax import struct
import jax
import jax.numpy as jnp


TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


@struct.dataclass
class EnvState:
    time: int


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 1


class Environment(Generic[TEnvState, TEnvParams]):  # object):
    """Jittable abstract base class for all gymnax Environments."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        action: Union[int, float, chex.Array],
        params: Optional[TEnvParams] = None,
    ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[TEnvParams] = None
    ) -> Tuple[chex.Array, TEnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        action: Union[int, float, chex.Array],
        params: TEnvParams,
    ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(
        self, key: chex.PRNGKey, params: TEnvParams
    ) -> Tuple[chex.Array, TEnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    @overload
    def get_obs(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @overload
    def get_obs(
        self,
        state: TEnvState,
    ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @overload
    def get_obs(
        self, state: TEnvState, key: chex.PRNGKey, params: TEnvParams
    ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def get_obs(
        self,
        state,
        params=None,
        key=None,
    ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state: TEnvState, params: TEnvParams) -> jnp.ndarray:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    def discount(self, state: TEnvState, params: TEnvParams) -> jnp.ndarray:
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

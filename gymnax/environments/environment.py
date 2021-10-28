import jax
import chex

from typing import Tuple, Union
from functools import partial

Array = chex.Array
PRNGKey = chex.PRNGKey


class Environment(object):
    """Jittable abstract base class for all gymnax Environments."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: PRNGKey, state: dict, action: Union[int, float], params: dict
    ) -> Tuple[Array, dict, float, bool]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_multimap(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self, key: PRNGKey, state: dict, action: Union[int, float], params: dict
    ) -> Tuple[Array, dict, float, bool]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Environment-specific reset."""
        raise NotImplementedError

    def get_obs(self, state: dict) -> Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state: dict) -> bool:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    def discount(self, state: dict, params: dict) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    @property
    def observation_space(self):
        """Observation space of the environment."""
        raise NotImplementedError

    @property
    def state_space(self):
        """State space of the environment."""
        raise NotImplementedError

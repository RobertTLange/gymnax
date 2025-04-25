"""DM env API wrapper for gymnax environment."""

import functools
from typing import TYPE_CHECKING, Optional, Union

import chex
import jax
import jax.numpy as jnp

from gymnax.environments import environment
from gymnax.wrappers import purerl

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class TimeStep:
    state: environment.EnvState
    reward: chex.Array
    discount: chex.Array
    observation: chex.Array


class GymnaxToDmEnvWrapper(purerl.GymnaxWrapper):
    """DM env API wrapper for gymnax environment."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> TimeStep:
        obs, state = self._env.reset(key, params)
        return TimeStep(state, jnp.array(0.0), jnp.array(1.0), obs)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        timestep: TimeStep,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> TimeStep:
        obs, state, reward, done, _ = self._env.step(
            key, timestep.state, action, params
        )
        return TimeStep(state, reward, 1.0 - done, obs)

"""DM env API wrapper for gymnax environment."""

import functools
from typing import TYPE_CHECKING

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

    def __init__(self, *, state, reward, discount, observation):
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "observation", observation)


class GymnaxToDmEnvWrapper(purerl.GymnaxWrapper):
    """DM env API wrapper for gymnax environment."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: environment.EnvParams | None = None
    ) -> TimeStep:
        obs, state = self._env.reset(key, params)
        return TimeStep(
            state=state, reward=jnp.array(0.0), discount=jnp.array(1.0), observation=obs
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        timestep: TimeStep,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> TimeStep:
        obs, state, reward, done, _ = self._env.step(
            key, timestep.state, action, params
        )
        return TimeStep(
            state=state, reward=reward, discount=1.0 - done, observation=obs
        )

"""DM env API wrapper for gymnax environment."""

from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gymnax.environments import environment
from gymnax.wrappers import purerl


@dataclass(frozen=True)
class TimeStep:
    state: environment.EnvState
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array

    def __init__(self, *, state, reward, discount, observation):
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "observation", observation)


class GymnaxToDmEnvWrapper(purerl.GymnaxWrapper):
    """DM env API wrapper for gymnax environment."""

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> TimeStep:
        obs, state = self._env.reset(key, params)
        return TimeStep(
            state=state, reward=jnp.array(0.0), discount=jnp.array(1.0), observation=obs
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
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

"""DM env API wrapper for gymnax environment."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment
from gymnax.wrappers import purerl


@struct.dataclass
class TimeStep:
    state: environment.EnvState
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array


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

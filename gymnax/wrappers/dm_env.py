import chex
from typing import Optional, Union
import jax
import jax.numpy as jnp
from functools import partial
from flax import struct
from gymnax.environments import environment, EnvState
from .purerl import GymnaxWrapper


@struct.dataclass
class TimeStep:
    state: EnvState
    reward: chex.Array
    discount: chex.Array
    observation: chex.Array


class GymnaxToDmEnvWrapper(GymnaxWrapper):
    """DM env API wrapper for gymnax environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> TimeStep:
        obs, state = self._env.reset(key, params)
        return TimeStep(state, 0.0, 1.0, obs)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        timestep: TimeStep,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> TimeStep:
        obs, state, reward, done, info = self._env.step(
            key, timestep.state, action, params
        )
        return TimeStep(state, reward, 1.0 - done, obs)

"""Wrapper exposing Gymnax environments through a dm_env-style API."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment
from gymnax.wrappers import purerl


@struct.dataclass
class TimeStep:
    """Transition returned by `GymnaxToDmEnvWrapper`.

    Attributes:
        state: Environment state, already reset if the episode ended.
        reward: Reward of the transition, 0 after a reset.
        discount: 0 if the episode terminated, 1 otherwise (including truncation).
        observation: Observation matching `state`.
        final_observation: Observation before any automatic reset.
    """

    state: environment.EnvState
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array
    final_observation: jax.Array


class GymnaxToDmEnvWrapper(purerl.GymnaxWrapper):
    """Return `TimeStep` objects from `reset` and `step`, as in dm_env.

    The wrapper stays functional and jittable: the environment state travels inside
    the `TimeStep`, which is passed back into `step`.

    Args:
        env (Environment): Environment to wrap.
    """

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> TimeStep:
        """Start a new episode and return its first `TimeStep`."""
        obs, state = self._env.reset(key, params)
        return TimeStep(
            state=state,
            reward=jnp.array(0.0),
            discount=jnp.array(1.0),
            observation=obs,
            final_observation=obs,
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        timestep: TimeStep,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> TimeStep:
        """Advance the environment from the state stored in `timestep`."""
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, timestep.state, action, params
        )
        return TimeStep(
            state=state,
            reward=reward,
            discount=1.0 - terminated,
            observation=obs,
            final_observation=info["final_observation"],
        )

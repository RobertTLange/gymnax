"""Utility wrapper to port gymnax env to evoJAX tasks."""

from typing import Tuple
import chex
from flax import struct
import gymnax
from gymnax.environments import environment
import jax

try:
    from evojax.task.base import VectorizedTask
    from evojax.task.base import TaskState
except Exception as exc:
    raise ImportError("You need to additionally install EvoJAX.") from exc


@struct.dataclass
class GymState(TaskState):
    state: environment.EnvState
    obs: chex.Array
    rng: chex.PRNGKey


class GymnaxToEvoJaxTask(VectorizedTask):
    """Task wrapper for gymnax environments."""

    def __init__(self, env_name: str, max_steps: int = 1000, test: bool = False):
        self.max_steps = max_steps
        self.test = test
        env, env_params = gymnax.make(env_name)
        env_params = env_params.replace(max_steps_in_episode=max_steps)
        self.obs_shape = env.obs_shape
        self.act_shape = env.num_actions
        self.num_actions = env.num_actions

        def reset_fn(key: chex.PRNGKey) -> GymState:
            key_re, key_ep = jax.random.split(key)
            obs, state = env.reset(key_re, env_params)
            state = GymState(state=state, obs=obs, rng=key_ep)
            return state

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(
            state: GymState, action: chex.Array
        ) -> Tuple[GymState, chex.Array, chex.Array]:
            key_st, key_ep = jax.random.split(state.rng)
            obs, env_state, reward, done, _ = env.step(
                key_st, state.state, action, env_params
            )
            state = state.replace(rng=key_ep, state=env_state, obs=obs)
            return state, reward, done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: chex.PRNGKey) -> GymState:
        return self._reset_fn(key)

    def step(self, state, action):  # -> Tuple[GymState, chex.Array, chex.Array]:
        return self._step_fn(state, action)

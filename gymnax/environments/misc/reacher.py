"""Reacher environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    angles: chex.Array
    angle_vels: chex.Array
    goal_xy: chex.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    torque_scale: float = 1.0
    dt: float = 0.05
    max_steps_in_episode: int = 100  # Steps in an episode (constant goal)


class Reacher(environment.Environment[EnvState, EnvParams]):
    """Reacher environment.


    Adapted from: https://github.com/unifyai/gym/blob/master/ivy_gym/reacher.py
    """

    def __init__(self, num_joints: int = 2):
        super().__init__()
        self.num_joints = num_joints

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Sample bernoulli reward, increase counter, construct input."""
        angle_accs = params.torque_scale * action
        angle_vels = state.angle_vels + params.dt * angle_accs
        angles = state.angles + params.dt * angle_vels

        state = EnvState(
            angles=angles,
            angle_vels=angle_vels,
            goal_xy=state.goal_xy,
            time=state.time + 1,
        )

        x = jnp.sum(jnp.cos(state.angles), axis=-1)
        y = jnp.sum(jnp.sin(state.angles), axis=-1)
        xy = jnp.concatenate(
            [jnp.expand_dims(x, axis=0), jnp.expand_dims(y, axis=0)], axis=0
        )
        reward = jnp.reshape(
            jnp.exp(-1 * jnp.sum((xy - state.goal_xy) ** 2, axis=-1)), (-1,)
        )
        reward = reward.squeeze()

        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Sample reward function + construct state as concat with timestamp
        rng_angle, rng_angle_v, rng_goal = jax.random.split(key, 3)

        angles = jax.random.uniform(
            rng_angle, minval=-jnp.pi, maxval=jnp.pi, shape=(self.num_joints,)
        )
        angle_vels = jax.random.uniform(
            rng_angle_v, minval=-1, maxval=1, shape=(self.num_joints,)
        )
        goal_xy = jax.random.uniform(
            rng_goal,
            minval=-self.num_joints,
            maxval=self.num_joints,
            shape=(2,),
        )

        state = EnvState(
            angles=angles,
            angle_vels=angle_vels,
            goal_xy=goal_xy,
            time=0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Concatenate reward, one-hot action and time stamp."""
        ob = (
            jnp.reshape(jnp.cos(state.angles), (1, self.num_joints)),
            jnp.reshape(jnp.sin(state.angles), (1, self.num_joints)),
            jnp.reshape(state.angle_vels, (1, self.num_joints)),
            jnp.reshape(state.goal_xy, (1, 2)),
        )
        ob = jnp.concatenate(ob, axis=0)
        return jnp.reshape(ob, (-1,))

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Reacher-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.num_joints

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        # if params is None:
        #   params = self.default_params
        low = jnp.array(self.num_joints * [-1], dtype=jnp.float32)
        high = jnp.array(self.num_joints * [1], dtype=jnp.float32)
        return spaces.Box(low, high, (self.num_joints,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            (self.num_joints * 3 + 2) * [-jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        high = jnp.array(
            (self.num_joints * 3 + 2) * [jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (self.num_joints * 3 + 2,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "angles": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (self.num_joints,),
                    jnp.float32,
                ),
                "angle_vels": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (self.num_joints),
                    jnp.float32,
                ),
                "goal_xy": spaces.Box(
                    -self.num_joints,
                    self.num_joints,
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

"""JAX implementation of a planar multi-joint reacher environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the Reacher environment.

    Attributes:
        angles: Array of shape `(num_joints,)` with the angle of each link, in
            radians.
        angle_vels: Array of shape `(num_joints,)` with the angular velocity of
            each link.
        goal_xy: Position of the goal as `(x, y)`.
        time: Number of steps taken in the current episode.
    """

    angles: jax.Array
    angle_vels: jax.Array
    goal_xy: jax.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the Reacher environment.

    Attributes:
        torque_scale: Factor by which the action is scaled to obtain the angular
            accelerations.
        dt: Duration of one step, used for Euler integration.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    torque_scale: float = 1.0
    dt: float = 0.05
    max_steps_in_episode: int = 100


class Reacher(environment.Environment[EnvState, EnvParams]):
    """Move the tip of a planar arm to a goal position by applying joint torques.

    The arm consists of `num_joints` links of unit length. The action, scaled by
    `params.torque_scale`, sets the angular acceleration of every link, which is
    integrated with step size `params.dt`. The tip position is the sum of the link
    vectors `(cos(angle), sin(angle))`. At every reset the angles are sampled
    uniformly in `[-pi, pi]`, the angular velocities in `[-1, 1]` and both goal
    coordinates in `[-num_joints, num_joints]`. The goal stays fixed for the whole
    episode. Supports differentiable transitions via `with_transition_gradients()`.

    - Observation: `Box` of shape `(3 * num_joints + 2,)` with the cosines and
      sines of the angles, the angular velocities and the goal position.
    - Actions: `Box` of shape `(num_joints,)` with bounds `[-1, 1]`. The torque
      applied to each joint.
    - Reward: `exp(-d**2)`, where `d` is the distance between the tip of the arm
      and the goal after the step.
    - Termination: never.
    - Truncation: after `params.max_steps_in_episode` steps (100 by default).

    Args:
        num_joints: Number of joints of the arm. Only the default of 2 is
            supported; other values fail when the observation is built.

    References:
        - [ivy_gym implementation](https://github.com/unifyai/gym/blob/master/ivy_gym/reacher.py)
    """

    _supports_transition_gradients = True

    def __init__(self, num_joints: int = 2):
        super().__init__()
        self.num_joints = num_joints

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Reacher-misc."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Integrate the joint dynamics and compute the goal distance reward."""
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
        observation, state = self._apply_transition_gradient_policy(
            self.get_obs(state, params), state
        )
        return (
            observation,
            state,
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling angles, velocities and goal."""
        # Sample reward function + construct state as concat with timestamp
        key_angle, key_angle_v, key_goal = jax.random.split(key, 3)

        angles = jax.random.uniform(
            key_angle, minval=-jnp.pi, maxval=jnp.pi, shape=(self.num_joints,)
        )
        angle_vels = jax.random.uniform(
            key_angle_v, minval=-1, maxval=1, shape=(self.num_joints,)
        )
        goal_xy = jax.random.uniform(
            key_goal,
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

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Concatenate angle cosines and sines, angular velocities and goal."""
        ob = (
            jnp.reshape(jnp.cos(state.angles), (1, self.num_joints)),
            jnp.reshape(jnp.sin(state.angles), (1, self.num_joints)),
            jnp.reshape(state.angle_vels, (1, self.num_joints)),
            jnp.reshape(state.goal_xy, (1, 2)),
        )
        ob = jnp.concatenate(ob, axis=0)
        return jnp.reshape(ob, (-1,))

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Reacher-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.num_joints

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
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

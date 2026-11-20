"""JAX implementation of a 2D semi-circle point robot environment."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the PointRobot environment.

    Attributes:
        last_action: Action passed to the previous step, before clipping (zeros
            after a reset).
        last_reward: Reward received in the previous step (0 after a reset).
        pos: Position of the robot as `(x, y)`.
        goal: Position of the goal as `(x, y)` on the semi-circle.
        goals_reached: Number of times the goal was reached in the current
            episode.
        time: Number of steps taken in the current episode.
    """

    last_action: jax.Array
    last_reward: jax.Array
    pos: jax.Array
    goal: jax.Array
    goals_reached: int
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the PointRobot environment.

    Attributes:
        max_force: Bound to which each action component is clipped (`+/-`).
        circle_radius: Radius of the semi-circle on which the goal is sampled.
        dense_reward: Whether the reward is the negative distance to the goal at
            every step instead of a sparse reward for being within the goal
            radius.
        goal_radius: Distance to the goal within which the goal counts as reached.
        center_init: Whether the robot starts at `(0, 0)` instead of a position
            sampled within the semi-circle.
        normalize_time: Whether the time step in the observation is rescaled so
            that steps 0 and 100 map to -1 and 1, instead of given as a raw step
            count.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    max_force: float = 0.1
    circle_radius: float = 1.0
    dense_reward: bool = False
    goal_radius: float = 0.2
    center_init: bool = False
    normalize_time: bool = True
    max_steps_in_episode: int = 100


class PointRobot(environment.Environment[EnvState, EnvParams]):
    """Navigate a 2D point robot to a hidden goal on a semi-circle.

    2D semi-circle point robot environment as in Dorfman et al. (2021). At every
    reset the goal is sampled uniformly on the upper semi-circle of radius
    `params.circle_radius` and stays fixed for the whole episode. The robot starts
    at a position sampled within the semi-circle, or at the origin if
    `params.center_init`. Each step adds the clipped action to the position. When
    the robot is within `params.goal_radius` of the goal at the start of a step,
    it is moved to a new start position at the end of that step. Supports
    differentiable transitions via `with_transition_gradients()`.

    - Observation: `Box` of shape `(6,)` with the robot position, the last reward,
      the last action and the time step (normalized if `params.normalize_time`).
    - Actions: `Box` of shape `(2,)` with the displacement in x and y, clipped to
      `[-params.max_force, params.max_force]`.
    - Reward: 1 if the robot is within `params.goal_radius` of the goal at the
      start of the step, otherwise 0. If `params.dense_reward`, the negative
      distance to the goal at the start of the step instead.
    - Termination: never.
    - Truncation: after `params.max_steps_in_episode` steps (100 by default).

    References:
        - [Dorfman et al. (2021)](https://openreview.net/pdf?id=IBdEfhLveS)
    """

    _supports_transition_gradients = True

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for PointRobot-misc."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Move the robot, compute the goal reward and construct the observation."""
        a = jnp.clip(action, -params.max_force, params.max_force)
        pos = state.pos + a
        goal_distance = jnp.linalg.norm(state.goal - state.pos)
        goal_reached = goal_distance <= params.goal_radius
        # Dense reward - distance to goal, sparse reward - 1 if in radius
        reward = jax.lax.select(params.dense_reward, -goal_distance, goal_reached * 1.0)
        sampled_pos = sample_agent_position(
            key, params.circle_radius, params.center_init
        )
        # Sample/set new initial position if goal was reached
        new_pos = jax.lax.select(goal_reached, sampled_pos, pos)
        state = EnvState(
            last_action=action,
            last_reward=reward,
            pos=new_pos,
            goal=state.goal,
            goals_reached=state.goals_reached + goal_reached,
            time=state.time + 1,
        )

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
        """Reset environment state by sampling goal and initial position."""
        # Sample reward function + construct state as concat with timestamp
        key_goal, key_pos = jax.random.split(key)
        angle = jax.random.uniform(key_goal, minval=0, maxval=jnp.pi)
        xs = params.circle_radius * jnp.cos(angle)
        ys = params.circle_radius * jnp.sin(angle)
        goal = jnp.array([xs, ys])
        sampled_pos = sample_agent_position(
            key_pos, params.circle_radius, params.center_init
        )

        state = EnvState(
            last_action=jnp.zeros(2),
            last_reward=jnp.array(0.0),
            pos=sampled_pos,
            goal=goal,
            goals_reached=0,
            time=0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Concatenate position, last reward, last action and time stamp."""
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        return jnp.hstack([state.pos, state.last_reward, state.last_action, time_rep])

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "PointRobot-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        low = jnp.array([-params.max_force, -params.max_force], dtype=jnp.float32)
        high = jnp.array([params.max_force, params.max_force], dtype=jnp.float32)
        return spaces.Box(low, high, (2,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            6 * [-jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        high = jnp.array(
            6 * [jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (6,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions),
                "last_reward": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def render(self, state: EnvState, params: EnvParams):
        """Plot the semi-circle, the goal region and the robot position."""
        fig, ax = plt.subplots()
        angles = jnp.linspace(0, jnp.pi, 100)
        x, y = jnp.cos(angles), jnp.sin(angles)
        ax.plot(x, y, color="k")
        plt.axis("scaled")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])

        circle = plt.Circle(
            (state.goal[0], state.goal[1]), radius=params.goal_radius, alpha=0.3
        )
        ax.add_artist(circle)

        circle = plt.Circle(
            (state.pos[0], state.pos[1]), radius=0.05, alpha=1, color="red"
        )
        ax.add_artist(circle)
        return fig, ax


def time_normalization(
    t: float, min_lim: float = -1.0, max_lim: float = 1.0, t_max: int = 100
) -> float:
    """Linearly rescale `t` so that 0 maps to `min_lim` and `t_max` to `max_lim`."""
    return (max_lim - min_lim) * t / t_max + min_lim


def sample_agent_position(
    key: jax.Array, circle_radius: float, center_init: bool
) -> jax.Array:
    """Sample a random position in the semi-circle (or set position to center)."""
    key_radius, key_angle = jax.random.split(key)
    sampled_radius = jax.random.uniform(key_radius, minval=0, maxval=circle_radius)
    sampled_angle = jax.random.uniform(key_angle, minval=0, maxval=jnp.pi)

    pos = jax.lax.select(
        center_init,
        jnp.zeros(2),
        jnp.array(
            [
                sampled_radius * jnp.cos(sampled_angle),
                sampled_radius * jnp.sin(sampled_angle),
            ]
        ),
    )
    return pos

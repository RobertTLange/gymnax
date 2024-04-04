"""Point Robot environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    last_action: chex.Array
    last_reward: jnp.ndarray
    pos: chex.Array
    goal: chex.Array
    goals_reached: int
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_force: float = 0.1  # Max action (+/-)
    circle_radius: float = 1.0  # Radius of semi-circle
    dense_reward: bool = False  # Distance reward at each timestep
    goal_radius: float = 0.2  # Radius for success
    center_init: bool = False  # Init at [0, 0]. Otherwise sample in radius
    normalize_time: bool = True  # Normalize timestep into [-1, 1]
    max_steps_in_episode: int = 100  # Steps in an episode (constant goal)


class PointRobot(environment.Environment[EnvState, EnvParams]):
    """2D Semi-Circle Point Robot environment similar to Dorfman et al.


    2021 https://openreview.net/pdf?id=IBdEfhLveS
    """

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
        rng_goal, rng_pos = jax.random.split(key)
        angle = jax.random.uniform(rng_goal, minval=0, maxval=jnp.pi)
        xs = params.circle_radius * jnp.cos(angle)
        ys = params.circle_radius * jnp.sin(angle)
        goal = jnp.array([xs, ys])
        sampled_pos = sample_agent_position(
            rng_pos, params.circle_radius, params.center_init
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

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Concatenate reward, one-hot action and time stamp."""
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        return jnp.hstack([state.pos, state.last_reward, state.last_action, time_rep])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "PointRobot-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
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
        """Small utility for plotting the agent's state."""
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
    """Normalize time integer into range given max time."""
    return (max_lim - min_lim) * t / t_max + min_lim


def sample_agent_position(
    key: chex.PRNGKey, circle_radius: float, center_init: bool
) -> chex.Array:
    """Sample a random position in circle (or set position to center)."""
    rng_radius, rng_angle = jax.random.split(key)
    sampled_radius = jax.random.uniform(rng_radius, minval=0, maxval=circle_radius)
    sampled_angle = jax.random.uniform(rng_angle, minval=0, maxval=jnp.pi)

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

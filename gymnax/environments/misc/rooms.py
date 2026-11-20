"""JAX implementation of the Four Rooms grid world of Sutton et al. (1999)."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the FourRooms environment.

    Attributes:
        pos: Agent position as `(row, column)` in the map.
        goal: Goal position as `(row, column)` in the map.
        time: Number of steps taken in the current episode.
    """

    pos: jax.Array
    goal: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the FourRooms environment.

    Attributes:
        fail_prob: Probability that one of the three other actions is executed
            instead of the chosen one. Implemented by replacing the action with a
            uniformly random one with probability `fail_prob * 4 / 3`.
        resample_init_pos: Whether the start position is sampled at every reset
            instead of using the fixed start position.
        resample_goal_pos: Whether the goal position is sampled at every reset
            instead of using the fixed goal position.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    fail_prob: float = 1.0 / 3
    resample_init_pos: bool = False
    resample_goal_pos: bool = False
    max_steps_in_episode: int = 500


four_rooms_map = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""


def string_to_bool_map(str_map: str) -> jax.Array:
    """Convert string map into a boolean map in which `True` marks open cells."""
    bool_map = []
    for row in str_map.split("\n")[1:]:
        bool_map.append([r == " " for r in row])
    return jnp.array(bool_map)


class FourRooms(environment.Environment[EnvState, EnvParams]):
    """Navigate through four connected rooms to a goal cell.

    Four Rooms grid world as in Sutton et al. (1999), comparable to
    `gym-fourrooms`. The 13 x 13 map consists of four rooms that are connected by
    single-cell doorways. Moving into a wall leaves the position unchanged, and
    with probability `params.fail_prob` a different action than the chosen one is
    executed. Since gymnax automatically resets the environment when an episode
    ends, the reset variants are selected via the parameters: start and goal are
    either fixed (`pos_fixed`, `goal_fixed`) or, if `params.resample_init_pos` or
    `params.resample_goal_pos` is set, sampled uniformly from the open cells at
    every reset. A sampled start never coincides with the goal.

    - Observation: `Box` of shape `(4,)` with the row and column of the agent
      followed by the row and column of the goal. If `use_visual_obs`, a `Box` of
      shape `(13, 13, 2)` with a wall map (1 for walls) and a one-hot map of the
      agent position. The agent map is indexed as `[column, row]`, transposed
      relative to the wall map.
    - Actions: `Discrete(4)`. 0 moves up, 1 right, 2 down and 3 left.
    - Reward: 1 for reaching the goal, otherwise 0.
    - Termination: the agent reaches the goal.
    - Truncation: after `params.max_steps_in_episode` steps (500 by default).

    Args:
        use_visual_obs: Whether to return the image observation instead of the
            agent and goal coordinates.
        goal_fixed: Goal position as `[row, column]` that is used unless
            `params.resample_goal_pos` is set. Defaults to `[8, 9]`.
        pos_fixed: Start position as `[row, column]` that is used unless
            `params.resample_init_pos` is set. Defaults to `[4, 1]`.

    References:
        - [Sutton et al. (1999)](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
        - [gym-fourrooms implementation](https://github.com/howardh/gym-fourrooms)
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        goal_fixed: list[int] | None = None,
        pos_fixed: list[int] | None = None,
    ):
        super().__init__()
        self.env_map = string_to_bool_map(four_rooms_map)
        self.occupied_map = 1 - self.env_map
        coords = []
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if self.env_map[y, x]:  # If it's an open space
                    coords.append([y, x])
        self.coords = jnp.array(coords)
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        # Any open space in the map can be a goal for the agent
        self.available_goals = self.coords

        # Whether to use 3D visual observation
        # Channel ID 0 - Wall (1) or not occupied (0)
        # Channel ID 1 - Agent location in maze
        self.use_visual_obs = use_visual_obs

        # Set fixed goal and position if we dont resample each time
        if goal_fixed is None:
            goal_fixed = [8, 9]
        if pos_fixed is None:
            pos_fixed = [4, 1]
        self.goal_fixed = jnp.array(goal_fixed)
        self.pos_fixed = jnp.array(pos_fixed)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for FourRooms-misc."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        key_random, key_action = jax.random.split(key)
        # Sample whether to choose a random action
        choose_random = jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )

        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state.pos)
        reward = jnp.logical_and(
            new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        ).astype(jnp.float32)

        # Update state dict and evaluate termination conditions
        state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state using fixed or sampled start and goal positions."""
        # Reset both the agents position and the goal location
        key_goal, key_pos = jax.random.split(key, 2)
        goal_new = reset_goal(key_goal, self.available_goals, params)
        # Only use resampled position if specified in EnvParams
        goal = jax.lax.select(params.resample_goal_pos, goal_new, self.goal_fixed)

        pos_new = reset_pos(key_pos, self.coords, goal)
        pos = jax.lax.select(params.resample_init_pos, pos_new, self.pos_fixed)
        state = EnvState(pos=pos, goal=goal, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return agent and goal coordinates or the image observation."""
        if not self.use_visual_obs:
            return jnp.array(
                [
                    state.pos[0],
                    state.pos[1],
                    state.goal[0],
                    state.goal[1],
                ]
            )
        else:
            agent_map = jnp.zeros(self.occupied_map.shape)
            agent_map = agent_map.at[state.pos[1], state.pos[0]].set(1)
            obs_array = jnp.stack([self.occupied_map, agent_map], axis=2)
            return obs_array

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check if agent has found the goal
        done_goal = jnp.logical_and(
            state.pos[0] == state.goal[0],
            state.pos[1] == state.goal[1],
        )
        return done_goal

    @property
    def name(self) -> str:
        """Environment name."""
        return "FourRooms-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if self.use_visual_obs:
            return spaces.Box(0, 1, (13, 13, 2), jnp.float32)
        else:
            return spaces.Box(
                jnp.min(self.coords), jnp.max(self.coords), (4,), jnp.float32
            )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(
                    jnp.min(self.coords),
                    jnp.max(self.coords),
                    (2,),
                    jnp.float32,
                ),
                "goal": spaces.Box(
                    jnp.min(self.coords),
                    jnp.max(self.coords),
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def render(self, state: EnvState, _: EnvParams):
        """Plot the map with the agent (A) and the goal (G)."""

        fig, ax = plt.subplots()
        ax.imshow(self.occupied_map, cmap="Greys")
        ax.annotate(
            "A",
            fontsize=20,
            xy=(state.pos[1], state.pos[0]),
            xycoords="data",
            xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
        )
        ax.annotate(
            "G",
            fontsize=20,
            xy=(state.goal[1], state.goal[0]),
            xycoords="data",
            xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


def reset_goal(key: jax.Array, available_goals: jax.Array, _: EnvParams) -> jax.Array:
    """Sample a goal position uniformly from the available goals."""
    goal_index = jax.random.randint(key, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(key: jax.Array, coords: jax.Array, goal: jax.Array) -> jax.Array:
    """Sample an agent position uniformly from the coordinates other than the goal."""
    pos_index = jax.random.randint(key, (), 0, coords.shape[0] - 1)
    collision = jnp.logical_and(
        coords[pos_index][0] == goal[0], coords[pos_index][1] == goal[1]
    )
    pos_index = jax.lax.select(collision, coords.shape[0] - 1, pos_index)
    return coords[pos_index][:]

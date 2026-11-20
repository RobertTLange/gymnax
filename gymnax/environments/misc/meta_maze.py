"""JAX implementation of the meta-maze environment of Miconi et al. (2019)."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the MetaMaze environment.

    Attributes:
        last_action: Action taken in the previous step (0 after a reset).
        last_reward: Reward received in the previous step (0 after a reset).
        pos: Agent position as `(row, column)` in the padded maze map.
        goal: Goal position as `(row, column)` in the padded maze map.
        time: Number of steps taken in the current episode.
    """

    last_action: int
    last_reward: jax.Array
    pos: jax.Array
    goal: jax.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the MetaMaze environment.

    The default values are inspired by the
    [SimpleMaze implementation](https://github.com/uber-research/backpropamine/blob/180c9101fa5be5a2da205da3399a92773d395091/simplemaze/maze.py#L414-L431).

    Attributes:
        reward: Reward for reaching the goal.
        punishment: Value added to the reward when the agent moves into a wall.
        normalize_time: Whether the time step in the observation is rescaled so
            that steps 0 and 100 map to -1 and 1, instead of given as a raw step
            count.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    reward: float = 10.0
    punishment: float = 0.0
    normalize_time: bool = False
    max_steps_in_episode: int = 200


def generate_maze_layout(maze_size: int, rf_size: int) -> jax.Array:
    """Generate a boolean maze map in which `True` marks walkable cells."""
    # Need to add wall offset if receptive field size is large
    rf_offset = int((rf_size - 1) / 2)

    # Need to add surrounding outer walls - first row
    maze = rf_offset * [(maze_size + 2 * rf_offset) * "x"]

    # Add inidividual rows with walls
    row_with_walls = (
        rf_offset * "x" + int((maze_size + 1) / 2) * " x" + (rf_offset - 1) * "x"
    )
    row_without_walls = rf_offset * "x" + maze_size * " " + rf_offset * "x"
    for r in range(maze_size):
        if r % 2 == 0:
            maze.append(row_without_walls)
        else:
            maze.append(row_with_walls)
    # Need to add surrounding outer walls - last row
    for _ in range(rf_offset):
        maze.append((maze_size + 2 * rf_offset) * "x")

    # Transform into boolean array map
    bool_map = []
    for row in maze:
        bool_map.append([r == " " for r in row])
    return jnp.array(bool_map)


class MetaMaze(environment.Environment[EnvState, EnvParams]):
    """Repeatedly find a hidden goal in a grid maze from local observations.

    Meta-learning maze as in Miconi et al. (2019). The maze is a square grid in
    which every second cell of every second row is a wall, surrounded by an outer
    wall. The goal is sampled uniformly from all walkable cells at every reset and
    stays fixed for the whole episode. The agent starts at `center_position`, the
    centre of the maze for the default `rf_size` (other values shift it; `rf_size = 5`
    places it on a wall), and is moved to a uniformly sampled
    walkable cell each time it reaches the goal. Moving into a wall leaves the
    position unchanged.

    - Observation: `Box` of shape `(rf_size**2 + 6,)` with the flattened
      `rf_size x rf_size` wall map around the agent (1 for walls), the one-hot
      encoded last action, the last reward and the time step (normalized if
      `params.normalize_time`).
    - Actions: `Discrete(4)`. 0 moves up, 1 right, 2 down and 3 left.
    - Reward: `params.reward` for reaching the goal, plus `params.punishment` for
      moving into a wall.
    - Termination: the agent's position equals the goal at the end of a step. As
      the agent is moved on reaching the goal, this only happens if the newly
      sampled position is the goal cell itself.
    - Truncation: after `params.max_steps_in_episode` steps (200 by default).

    Args:
        maze_size: Number of rows and columns of the maze without the outer wall.
            Must be odd.
        rf_size: Side length of the square receptive field observed around the
            agent. Must be odd and larger than 1.

    References:
        - [Miconi et al. (2019)](https://arxiv.org/abs/2002.10585)
        - [SimpleMaze implementation](https://github.com/uber-research/backpropamine/blob/master/simplemaze/maze.py)
    """

    def __init__(self, maze_size: int = 9, rf_size: int = 3):
        super().__init__()
        # Maze size and receptive field have to be uneven (centering)
        assert maze_size % 2 != 0
        assert rf_size % 2 != 0 and rf_size > 1
        self.maze_size = maze_size
        self.rf_size = rf_size
        # Offset of walls top/bottom and left/right
        self.rf_off = jnp.int32((self.rf_size - 1) / 2)
        # Generate the maze layout
        self.env_map = generate_maze_layout(maze_size, rf_size)
        center = jnp.int32((self.env_map.shape[0] - 1) / 2 + self.rf_off - 1)
        self.center_position = jnp.array([center, center])
        self.occupied_map = 1 - self.env_map
        coords = []
        # Get all walkable positions or positions that can be goals
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if self.env_map[y, x]:
                    coords.append([y, x])
        self.coords = jnp.array(coords)
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        # Any open space in the map can be a goal for the agent
        self.available_goals = self.coords

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for MetaMaze-misc."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state.pos)
        goal_reached = jnp.logical_and(
            new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        )
        reward = (
            goal_reached * params.reward  # Add goal reward
            + (1 - in_map) * params.punishment  # Add punishment for wall
        )

        # Sample a new starting position for case when goal is reached
        pos_sampled = reset_pos(key, self.coords)
        new_pos = jax.lax.select(goal_reached, pos_sampled, new_pos)
        # Update state dict and evaluate termination conditions
        state = EnvState(
            last_action=action,
            last_reward=reward,
            pos=new_pos,
            goal=state.goal,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling a new goal position."""
        # Reset both the agents position and the goal location
        goal = reset_goal(key, self.available_goals, params)
        # Initialize agent always at fixed center position of maze
        state = EnvState(
            last_action=0,
            last_reward=jnp.array(0.0),
            pos=self.center_position,
            goal=goal,
            time=0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Concatenate local wall map, one-hot action, reward and time stamp."""
        rf_obs = jax.lax.dynamic_slice(
            self.occupied_map,
            (state.pos[0] - self.rf_off, state.pos[1] - self.rf_off),
            (self.rf_size, self.rf_size),
        ).reshape(-1)
        action_one_hot = jax.nn.one_hot(state.last_action, self.num_actions).squeeze()
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        return jnp.hstack([rf_obs, action_one_hot, state.last_reward, time_rep])

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check if agent has found the goal
        done_goal = jnp.logical_and(
            state.pos[0] == state.goal[0],
            state.pos[1] == state.goal[1],
        )
        return done_goal

    def render(self, state: EnvState, _: EnvParams):
        """Plot the maze with the agent (A) and the goal (G)."""

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

    @property
    def name(self) -> str:
        """Environment name."""
        return "MetaMaze-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            self.rf_size**2 * [0] + self.num_actions * [0] + [0, 0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            self.rf_size**2 * [1]
            + self.num_actions * [1]
            + [1, params.max_steps_in_episode],
            dtype=jnp.float32,
        )
        return spaces.Box(
            low, high, (self.rf_size**2 + self.num_actions + 2,), jnp.float32
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions),
                "last_reward": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
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


def reset_goal(key: jax.Array, available_goals: jax.Array, _: EnvParams) -> jax.Array:
    """Sample a goal position uniformly from the available goals."""
    goal_index = jax.random.randint(key, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(key: jax.Array, coords: jax.Array) -> jax.Array:
    """Sample an agent position uniformly from the walkable coordinates."""
    pos_index = jax.random.randint(key, (), 0, coords.shape[0])
    return coords[pos_index][:]


def time_normalization(
    t: float, min_lim: float = -1.0, max_lim: float = 1.0, t_max: int = 100
) -> float:
    """Linearly rescale `t` so that 0 maps to `min_lim` and `t_max` to `max_lim`."""
    return (max_lim - min_lim) * t / t_max + min_lim

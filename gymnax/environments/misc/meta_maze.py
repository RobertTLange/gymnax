"""JAX Compatible version of meta-maze environment (Miconi et al., 2019).


Source: Comparable to
github.com/uber-research/backpropamine/blob/master/simplemaze/maze.py
"""

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
    last_action: int
    last_reward: jnp.ndarray
    pos: chex.Array
    goal: chex.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    # github.com/uber-research/backpropamine/blob/180c9101fa5be5a2da205da3399a92773d395091/simplemaze/maze.py#L414-L431
    reward: float = 10.0
    punishment: float = 0.0
    normalize_time: bool = False
    max_steps_in_episode: int = 200


def generate_maze_layout(maze_size: int, rf_size: int) -> chex.Array:
    """Generate array representation of maze layout with walls."""
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
    """JAX Compatible version of meta-maze environment (Miconi et al., 2019)."""

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
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
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

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
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

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        # Check if agent has found the goal
        done_goal = jnp.logical_and(
            state.pos[0] == state.goal[0],
            state.pos[1] == state.goal[1],
        )
        done = jnp.logical_or(done_goal, done_steps)
        return done

    def render(self, state: EnvState, _: EnvParams):
        """Small utility for plotting the agent's state."""

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

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
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


def reset_goal(
    rng: chex.PRNGKey, available_goals: chex.Array, _: EnvParams
) -> chex.Array:
    """Reset the goal state/position in the environment."""
    goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(rng: chex.PRNGKey, coords: chex.Array) -> chex.Array:
    """Reset the position of the agent."""
    pos_index = jax.random.randint(rng, (), 0, coords.shape[0])
    return coords[pos_index][:]


def time_normalization(
    t: float, min_lim: float = -1.0, max_lim: float = 1.0, t_max: int = 100
) -> float:
    """Normalize time integer into range given max time."""
    return (max_lim - min_lim) * t / t_max + min_lim

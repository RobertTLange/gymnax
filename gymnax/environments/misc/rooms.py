import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple
import chex
from flax import struct


@struct.dataclass
class EnvState:
    pos: chex.Array
    goal: chex.Array
    time: int


@struct.dataclass
class EnvParams:
    fail_prob: float = 1.0 / 3
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


def string_to_bool_map(str_map: str) -> chex.Array:
    """Convert string map into boolean walking map."""
    bool_map = []
    for row in str_map.split("\n")[1:]:
        bool_map.append([r == " " for r in row])
    return jnp.array(bool_map)


class FourRooms(environment.Environment):
    """
    JAX Compatible version of Four Rooms environment (Sutton et al., 1999).
    Source: Comparable to https://github.com/howardh/gym-fourrooms
    Since gymnax automatically resets env at done, we abstract different resets
    """

    def __init__(self):
        super().__init__()
        self.env_map = string_to_bool_map(four_rooms_map)
        coords = []
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if self.env_map[y, x]:  # If it's an open space
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
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        key_random, key_action = jax.random.split(key)
        # Sample whether to choose a random action
        choose_random = (
            jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
        )
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )

        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state.pos)
        reward = jnp.logical_and(
            new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        )

        # Update state dict and evaluate termination conditions
        state = EnvState(new_pos, state.goal, state.time + 1)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
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
        rng_goal, rng_pos = jax.random.split(key, 2)
        goal = reset_goal(rng_goal, self.available_goals, params)
        pos = reset_pos(rng_pos, self.coords, goal)
        state = EnvState(pos, goal, 0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.array(
            [
                state.pos[0],
                state.pos[1],
                state.goal[0],
                state.goal[1],
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time > params.max_steps_in_episode
        # Check if agent has found the goal
        done_goal = jnp.logical_and(
            state.pos[0] == state.goal[0],
            state.pos[1] == state.goal[1],
        )
        done = jnp.logical_or(done_goal, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "FourRooms-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
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


def reset_goal(
    rng: chex.PRNGKey, available_goals: chex.Array, params: EnvParams
) -> chex.Array:
    """Reset the goal state/position in the environment."""
    goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(
    rng: chex.PRNGKey, coords: chex.Array, goal: chex.Array
) -> chex.Array:
    """Reset the position of the agent."""
    pos_index = jax.random.randint(rng, (), 0, coords.shape[0] - 1)
    collision = jnp.logical_and(
        coords[pos_index][0] == goal[0], coords[pos_index][1] == goal[1]
    )
    pos_index = jax.lax.select(collision, coords.shape[0] - 1, pos_index)
    return coords[pos_index][:]

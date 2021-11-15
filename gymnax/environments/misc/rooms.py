import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


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


def string_to_bool_map(str_map: str):
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

        # Any open space in the map can be a goal for the agent
        self.available_goals = self.coords

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "fail_prob": 1 / 3,
            "max_steps_in_episode": 500,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        key_random, key_action = jax.random.split(key)
        choose_random = jax.random.uniform(key_random, ()) < params["fail_prob"] * 4 / 3
        action = jax.lax.select(
            choose_random, self.action_space.sample(key_action), action
        )
        directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        p = state["pos"] + directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state["pos"])
        reward = jnp.logical_and(
            new_pos[0] == state["goal"][0], new_pos[1] == state["goal"][1]
        )

        # Update state dict and evaluate termination conditions
        state = {
            "pos": new_pos,
            "goal": state["goal"],
            "time": state["time"] + 1,
        }
        done = self.is_terminal(state, params)
        state["terminal"] = done
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location
        rng_goal, rng_pos = jax.random.split(key, 2)
        goal = reset_goal(rng_goal, self.available_goals, params)
        pos = reset_pos(rng_pos, self.coords, goal)
        state = {
            "pos": pos,
            "goal": goal,
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        return jnp.array(
            [state["pos"][0], state["pos"][1], state["goal"][0], state["goal"][1]]
        )

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state["time"] > params["max_steps_in_episode"]
        done_goal = jnp.logical_and(
            state["pos"][0] == state["goal"][0], state["pos"][1] == state["goal"][1]
        )
        done = jnp.logical_or(done_goal, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "FourRooms-misc"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(jnp.min(self.coords), jnp.max(self.coords), (4,), jnp.float32)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(
                    jnp.min(self.coords), jnp.max(self.coords), (2,), jnp.float32
                ),
                "goal": spaces.Box(
                    jnp.min(self.coords), jnp.max(self.coords), (2,), jnp.float32
                ),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )


def reset_goal(rng: PRNGKey, available_goals: Array, params: dict):
    """Reset the goal state/position in the environment."""
    goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(rng: PRNGKey, coords: Array, goal: Array):
    """Reset the position of the agent."""
    pos_index = jax.random.randint(rng, (), 0, coords.shape[0] - 1)
    collision = jnp.logical_and(
        coords[pos_index][0] == goal[0], coords[pos_index][1] == goal[1]
    )
    pos_index = jax.lax.select(collision, coords.shape[0] - 1, pos_index)
    return coords[pos_index][:]

import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class Catch(environment.Environment):
    """
    JAX Compatible version of Catch bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py
    """

    def __init__(self, rows: int = 10, columns: int = 5):
        super().__init__()
        self.rows = rows
        self.columns = columns

    @property
    def default_params(self):
        # Default environment parameters
        return {"max_steps_in_episode": 100000000}

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        # Sample new init state each step & use if there was a reset!
        ball_x, ball_y, paddle_x, paddle_y = sample_init_state(
            key, self.rows, self.columns
        )
        prev_done = state["prev_done"]

        # Move the paddle + drop the ball.
        dx = action - 1  # [-1, 0, 1] = Left, no-op, right.
        paddle_x = (
            jnp.clip(state["paddle_x"] + dx, 0, self.columns - 1) * (1 - prev_done)
            + paddle_x * prev_done
        )
        ball_y = (state["ball_y"] + 1) * (1 - prev_done) + ball_y * prev_done
        ball_x = state["ball_x"] * (1 - prev_done) + ball_x * prev_done
        paddle_y = state["paddle_y"] * (1 - prev_done) + paddle_y * prev_done

        # Rewrite reward as boolean multiplication
        prev_done = ball_y == paddle_y
        catched = paddle_x == ball_x
        reward = prev_done * (1.0 * catched + -1.0 * (1 - catched))

        state = {
            "ball_x": ball_x,
            "ball_y": ball_y,
            "paddle_x": paddle_x,
            "paddle_y": paddle_y,
            "prev_done": prev_done,
            "time": state["time"] + 1,
        }

        # Check number of steps in episode termination condition
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
        ball_x, ball_y, paddle_x, paddle_y = sample_init_state(
            key, self.rows, self.columns
        )
        # Last two state vector correspond to timestep and done
        state = {
            "ball_x": ball_x,
            "ball_y": ball_y,
            "paddle_x": paddle_x,
            "paddle_y": paddle_y,
            "prev_done": False,
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((self.rows, self.columns))
        obs = jax.ops.index_update(
            obs, jax.ops.index[state["ball_y"], state["ball_x"]], 1.0
        )
        obs = jax.ops.index_update(
            obs, jax.ops.index[state["paddle_y"], state["paddle_x"]], 1.0
        )
        return obs

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done_loose = state["ball_y"] == self.rows - 1
        done_steps = state["time"] > params["max_steps_in_episode"]
        done = jnp.logical_or(done_loose, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Catch-bsuite"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.rows, self.columns), dtype=jnp.int_)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "ball_x": spaces.Discrete(self.columns),
                "ball_y": spaces.Discrete(self.rows),
                "paddle_x": spaces.Discrete(self.columns),
                "paddle_y": spaces.Discrete(self.rows),
                "prev_done": spaces.Discrete(2),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )


def sample_init_state(key, rows: int, columns: int):
    """Sample a new initial state."""
    # high = jnp.zeros((params["rows"], params["columns"]))
    ball_x = jax.random.randint(key, shape=(), minval=0, maxval=columns)
    ball_y = 0
    paddle_x = columns // 2
    paddle_y = rows - 1
    return ball_x, ball_y, paddle_x, paddle_y

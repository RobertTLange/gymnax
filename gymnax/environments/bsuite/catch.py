import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey


class Catch(environment.Environment):
    """
    JAX Compatible version of Catch bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py
    """
    def __init__(self):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"max_steps_in_episode": 2000,
                                      "rows": 10,
                                      "columns": 5})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        # Sample new init state each step & use if there was a reset!
        ball_x, ball_y, paddle_x, paddle_y = sample_init_state(key,
                                                               self.env_params)
        prev_done = state["prev_done"]

        # Move the paddle + drop the ball.
        dx = action - 1  # [-1, 0, 1] = Left, no-op, right.
        paddle_x = (jnp.clip(state["paddle_x"] + dx, 0,
                             self.env_params["columns"] - 1)
                    * (1-prev_done) + paddle_x * prev_done)
        ball_y = ((state["ball_y"] + 1) * (1-prev_done)
                   + ball_y * prev_done)
        ball_x = state["ball_x"] * (1-prev_done) + ball_x * prev_done
        paddle_y = (state["paddle_y"] * (1-prev_done)
                    + paddle_y * prev_done)

        # Rewrite reward as boolean multiplication
        prev_done = (ball_y == paddle_y)
        catched = (paddle_x == ball_x)
        reward = prev_done * (1 * catched + -1 * (1 - catched))

        state = {"ball_x": ball_x,
                 "ball_y": ball_y,
                 "paddle_x": paddle_x,
                 "paddle_y": paddle_y,
                 "prev_done": prev_done,
                 "time": state["time"] + 1}

        # Check number of steps in episode termination condition
        done = self.is_terminal(state)
        state["terminal"] = done
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done,
                {"discount": self.discount(state)})

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        ball_x, ball_y, paddle_x, paddle_y = sample_init_state(key,
                                                               self.env_params)
        # Last two state vector correspond to timestep and done
        state = {"ball_x": ball_x,
                 "ball_y": ball_y,
                 "paddle_x": paddle_x,
                 "paddle_y": paddle_y,
                 "prev_done": 0,
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        obs = jnp.zeros((self.env_params["rows"], self.env_params["columns"]))
        obs = jax.ops.index_update(obs, jax.ops.index[state["ball_y"],
                                                      state["ball_x"]], 1.)
        obs = jax.ops.index_update(obs, jax.ops.index[state["paddle_y"],
                                                      state["paddle_x"]], 1.)
        return obs

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done_loose = state["ball_y"] == self.env_params["rows"] - 1
        done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        done = jnp.logical_or(done_loose, done_steps)
        return done

    @property
    def name(self) -> str:
        """ Environment name. """
        return "Catch-bsuite"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(3)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(0, 1, (self.env_params["rows"],
                                 self.env_params["columns"]),
                          dtype=jnp.int_)

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"ball_x": spaces.Discrete(self.env_params["columns"]),
             "ball_y": spaces.Discrete(self.env_params["rows"]),
             "paddle_x": spaces.Discrete(self.env_params["columns"]),
             "paddle_y": spaces.Discrete(self.env_params["rows"]),
             "prev_done": spaces.Discrete(2),
             "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})


def sample_init_state(key, params):
    """ Sample a new initial state. """
    high = jnp.zeros((params["rows"], params["columns"]))
    ball_x = jax.random.randint(key, shape=(),
                                minval=0, maxval=params["columns"])
    ball_y = 0
    paddle_x = params["columns"] // 2
    paddle_y = params["rows"] - 1
    return ball_x, ball_y, paddle_x, paddle_y

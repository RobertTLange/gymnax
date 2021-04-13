import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict


# JAX Compatible version of Catch bsuite environment. Source:
# https://github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py

# Default environment parameters
params_catch = FrozenDict({"max_steps_in_episode": 2000,
                           "rows": 10,
                           "columns": 5})


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    # Sample new init state at each step and use only if there was a reset!
    ball_x, ball_y, paddle_x, paddle_y = sample_init_state(rng_input, params)
    prev_done = state["prev_done"]

    # Move the paddle + drop the ball.
    dx = action - 1  # [-1, 0, 1] = Left, no-op, right.
    state["paddle_x"] = (jnp.clip(state["paddle_x"] + dx, 0,
                                  params["columns"] - 1)
                         * (1-prev_done) + paddle_x * prev_done)
    state["ball_y"] = (state["ball_y"] + 1) * (1-prev_done) + ball_y * prev_done
    state["ball_x"] = state["ball_x"] * (1-prev_done) + ball_x * prev_done
    state["paddle_y"] = state["paddle_y"] * (1-prev_done) + paddle_y * prev_done

    # Rewrite reward as boolean multiplication
    state["prev_done"] = (ball_y == paddle_y)
    catched = (paddle_x == ball_x)
    reward = state["prev_done"] * (1 * catched + -1 * (1 - catched))

    # Check number of steps in episode termination condition
    state["time"] += 1
    done_steps = (state["time"] > params["max_steps_in_episode"])
    return get_obs(state, params), state, reward, done_steps, {}


def sample_init_state(rng_input, params):
    """ Sample a new initial state. """
    high = jnp.zeros((params["rows"], params["columns"]))
    ball_x = jax.random.randint(rng_input, shape=(),
                                minval=0, maxval=params["columns"])
    ball_y = 0
    paddle_x = params["columns"] // 2
    paddle_y = params["rows"] - 1
    return ball_x, ball_y, paddle_x, paddle_y


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    ball_x, ball_y, paddle_x, paddle_y = sample_init_state(rng_input, params)
    # Last two state vector correspond to timestep and done
    state = {"ball_x": ball_x,
             "ball_y": ball_y,
             "paddle_x": paddle_x,
             "paddle_y": paddle_y,
             "time": 0,
             "prev_done": 0}
    return get_obs(state, params), state


def get_obs(state, params):
    """ Return observation from raw state trafo. """
    board = jnp.zeros((params["rows"], params["columns"]))
    board = jax.ops.index_update(board, jax.ops.index[state["ball_y"],
                                                      state["ball_x"]], 1.)
    board = jax.ops.index_update(board, jax.ops.index[state["paddle_y"],
                                                      state["paddle_x"]], 1.)
    return board


reset_catch = jit(reset, static_argnums=(1,))
step_catch = jit(step, static_argnums=(1,))

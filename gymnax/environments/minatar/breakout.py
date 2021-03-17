import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Breakout MinAtar environment. Source:
# github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py


"""
- Player controls paddle on bottom of screen.
- Must bounce ball to break 3 rows if bricks along top of screen.
- A reward of +1 is given for each broken brick.
- If all bricks are cleared another 3 rows are added.
- Ball travels only along diagonals, when paddle/wall hit it bounces off
- Termination if ball hits bottom of screen.
- Ball direction is indicated by a trail channel.
- There is no difficulty increase.
- Channels are encoded as follows: 'paddle':0, 'ball':1, 'trail':2, 'brick':3
- Observation has dimensionality (10, 10, 4)
- Actions are encoded as follows: 'l': 0, 'r': 1.
- Note that is different from MinAtar where the action range is 6.
"""

# Default environment parameters
params_breakout = {}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    state = 0
    reward = 0
    done = False
    info = {}
    return get_obs(state), state, reward, done, info


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    ball_start = jax.random.choice(rng_input, jnp.array([0, 1]), shape=(1,))
    ball_x = jnp.array([0, 9])[ball_start]
    ball_dir = jnp.array([2, 3])[ball_start]
    brick_map = jnp.zeros((10, 10))
    brick_map = jax.ops.index_update(brick_map, jax.ops.index[1:4, :], 1)
    state = {
        "ball_y": 3,
        "ball_x": ball_x,
        "ball_dir": ball_dir,
        "pos": 4,
        "brick_map": brick_map,
        "strike": 0,
        "last_y": 3,
        "last_x": ball_x,
        "terminal": 0
    }
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    obs = jnp.zeros((10, 10, 4),dtype=bool)
    # Set the position of the ball, paddle, trail and the brick map
    obs = jax.ops.index_update(obs, jax.ops.index[state["ball_y"],
                                                  state["ball_x"],
                                                  1], 1)
    obs = jax.ops.index_update(obs, jax.ops.index[9, state["pos"],
                                                  0], 1)
    obs = jax.ops.index_update(obs, jax.ops.index[state["last_y"],
                                                  state["last_x"],
                                                  2], 1)
    obs = jax.ops.index_update(obs, jax.ops.index[:, :, 3],
                               state["brick_map"])
    return obs


reset_breakout = jit(reset)
step_breakout = jit(step)

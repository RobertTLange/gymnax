import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Breakout MinAtar environment. Source:
# github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py


"""
ENVIRONMENT DESCRIPTION - 'Breakout-MinAtar'
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
- Actions are encoded as follows: ['n','l','u','r','d','f']
- Note that is different from MinAtar where the action range is 6.
"""

# Default environment parameters
params_breakout = {}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    reward = 0
    done = False
    info = {}

    state["pos"] = (jnp.maximum(0, state["pos"] - 1) * (action == 1)
                    + jnp.minimum(9, state["pos"] + 1) * (action == 3)
                    + state["pos"] * jnp.logical_and(action != 1,
                                                     action != 3))

    # Update ball position
    state["last_x"] = state["ball_x"]
    state["last_y"] = state["ball_y"]
    new_x = ((state["ball_x"] - 1) * (state["ball_dir"] == 0) +
             (state["ball_x"] + 1) * (state["ball_dir"] == 1) +
             (state["ball_x"] + 1) * (state["ball_dir"] == 2) +
             (state["ball_x"] - 1) * (state["ball_dir"] == 3))
    new_y = ((state["ball_y"] - 1) * (state["ball_dir"] == 0) +
             (state["ball_y"] - 1) * (state["ball_dir"] == 1) +
             (state["ball_y"] + 1) * (state["ball_dir"] == 2) +
             (state["ball_y"] + 1) * (state["ball_dir"] == 3))

    # Boundary conditions and brick map update
    strike_toggle = 0
    border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    new_x = (border_cond_x * (0 * (new_x < 0) + 9 * (new_x > 9))
             + (1 - border_cond_x) * new_x)
    # Reflect ball direction if bounced off at x border
    state["ball_dir"] = ((border_cond_x *
                         jnp.array([1, 0, 3, 2])[state["ball_dir"]])
                         + (1 - border_cond_x) * state["ball_dir"])

    # Reflect ball direction if bounced off at y border
    border_cond1_y = (new_y < 0)
    new_y = border_cond1_y * 0 + (1 - border_cond1_y) * new_y
    state["ball_dir"] = ((border_cond1_y *
                         jnp.array([3, 2, 1, 0])[state["ball_dir"]])
                         + (1 - border_cond1_y) * state["ball_dir"])

    # 1st NASTY ELIF BEGINS HERE...
    strike_toggle = jnp.logical_and(1 - border_cond1_y,
                                    state["brick_map"][new_y, new_x] == 1)
    strike_bool = jnp.logical_and((1 - state["strike"]), strike_toggle)

    reward += strike_bool
    state["strike"] = strike_bool
    state["brick_map"] = (strike_bool *
                          jax.ops.index_update(state["brick_map"],
                                               new_y, new_x, 0)
                          + (1 - strike_bool) * state["brick_map"])
    new_y = (strike_bool * state["last_y"] + (1 - strike_bool) * new_y)
    state["ball_dir"] = (strike_bool * jnp.array([3,2,1,0])[state["ball_dir"]]
                         + (1 - strike_bool) * state["ball_dir"])

    # 2nd NASTY ELIF BEGINS HERE...
    new_bricks = jnp.logical_and(1 - strike_toggle, new_y == 9)
    spawn_bricks = jnp.logical_and(new_bricks,
                                   jnp.count_nonzero(state["brick_map"]) == 0)
    state["brick_map"] = (spawn_bricks *
                          jax.ops.index_update(state["brick_map"],
                                               jax.ops.index[1:4, :], 1)
                          + (1 - spawn_bricks) * state["brick_map"])
    redirect_ball1 = jnp.logical_and(new_bricks,
                                     state["ball_x"] == state["pos"])
    state["ball_dir"] = (redirect_ball1 *
                         jnp.array([3, 2, 1, 0])[state["ball_dir"]]
                         + (1 - redirect_ball1) * state["ball_dir"])
    new_y = redirect_ball1 * state["last_y"] + (1 - redirect_ball1) * new_y

    redirect_ball2 = jnp.logical_and(1 - redirect_ball1, new_x == state["pos"])
    state["ball_dir"] = (redirect_ball2 *
                         jnp.array([2, 3, 0, 1])[state["ball_dir"]]
                         + (1 - redirect_ball2) * state["ball_dir"])
    new_y = redirect_ball2 * state["last_y"] + (1 - redirect_ball2) * new_y

    redirect_cond = jnp.logical_and(1 - redirect_ball1, 1 - redirect_ball2)
    terminal_cond = jnp.logical_and(new_bricks, redirect_cond)
    state["terminal"] = terminal_cond

    state["strike"] = (0 * (1 - strike_toggle) +
                       state["strike"] * strike_toggle)
    state["ball_x"] = new_x
    state["ball_y"] = new_y

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

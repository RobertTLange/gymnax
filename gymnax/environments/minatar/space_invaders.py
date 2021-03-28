import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Freeway MinAtar environment. Source:
# github.com/kenjyoung/MinAtar/blob/master/minatar/environments/space_invaders.py


"""
ENVIRONMENT DESCRIPTION - 'SpaceInvaders-MinAtar'
- Player controls cannon at bottom of screen and can shoot bullets at aliens
- Aliens move across screen until one of them hits the edge.
- At this point all move down and switch directions.
- Current alien dir indicated by 2 channels (left/right) - active at position.
- Reward of +1 is given each time alien is shot and alien is removed.
- Aliens will also shoot bullets back at player.
- Alien speed increases when only few of them are left.
- When only one alien is left, it will move at one cell per frame.
- When wave of aliens is cleared, slightly faster new one will spawn.
- Termination occurs when an alien or bullet hits the player.
- Channels are encoded as follows: 'cannon':0, 'alien':1, 'alien_left':2,
- 'alien_right':3, 'friendly_bullet':4, 'enemy_bullet':5
- Observation has dimensionality (10, 10, 6)
- Actions are encoded as follows: ['n','l','u','r','d','f']
- Only actions 1, 3 and 5 ('l' and 'r', 'f') lead to a change!
"""

# Default environment parameters
params_space_invaders = {"shot_cool_down": 5,
                         "enemy_move_interval": 12,
                         "enemy_shot_interval": 10}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    state = 0
    reward = 0
    done = False
    info = {}
    return get_obs(state), state, reward, done, info


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    alien_map = jnp.zeros((10, 10))
    alien_map = jax.ops.index_update(alien_map, jax.ops.index[0:4, 2:9], 1)
    state = {"pos": 5,
             "f_bullet_map": jnp.zeros((10, 10)),
             "e_bullet_map": jnp.zeros((10, 10)),
             "alien_map": alien_map,
             "alien_dir": -1,
             "enemy_move_interval": params["enemy_move_interval"],
             "alien_move_timer": params["enemy_move_interval"],
             "alien_shot_timer": params["enemy_shot_interval"],
             "ramp_index": 0,
             "shot_timer": 0,
             "terminal": False}

    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    obs = jnp.zeros((10, 10, 6), dtype=bool)
    # Update cannon, aliens - left + right dir, friendly + enemy bullet
    obs = jax.ops.index_update(obs, jax.ops.index[9, state["pos"], 0], 1)
    obs = jax.ops.index_update(obs, jax.ops.index[:, :, 1], state["alien_map"])
    left_dir_cond = (state["alien_dir"] < 0)
    obs = (left_dir_cond * jax.ops.index_update(obs, jax.ops.index[:, :, 2],
                                                state["alien_map"])
           + (1-left_dir_cond) * jax.ops.index_update(obs,
                                                      jax.ops.index[:, :, 3],
                                                      state["alien_map"]))
    obs = jax.ops.index_update(obs, jax.ops.index[:, :, 4],
                               state["f_bullet_map"])
    obs = jax.ops.index_update(obs, jax.ops.index[:, :, 5],
                               state["e_bullet_map"])
    return obs


reset_space_invaders = jit(reset)
step_space_invaders = jit(step)

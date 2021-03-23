import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Freeway MinAtar environment. Source:
# github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py


"""
ENVIRONMENT DESCRIPTION - 'Freeway-MinAtar'
- Player starts at bottom of screen and can travel up/down.
- Player speed is restricted s.t. player only moves every 3 frames.
- A reward +1 is given when player reaches top of screen -> returns to bottom.
- 8 cars travel horizontally on screen and teleport to other side at edge.
- When player is hit by a car, he is returned to the bottom of the screen.
- Car direction and speed are indicated by 5 trail channels.
- Each time player reaches top of screen, car speeds are randomized.
- Termination occurs after 2500 frames.
- Channels are encoded as follows: 'chicken':0, 'car':1, 'speed1':2,
- 'speed2':3, 'speed3':4, 'speed4':5, 'speed5':6
- Observation has dimensionality (10, 10, 4)
- Actions are encoded as follows: ['n','l','u','r','d','f']
- Only actions 2 and 4 ('u' and 'd') lead to a change!
"""

# Default environment parameters
params_freeway = {}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    reward = 0
    done = False
    info = {}
    return get_obs(state), state, reward, done, info


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    state = {"pos":,
             "cars": }
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    obs = jnp.zeros((10, 10, 7), dtype=bool)
    # Set the position of the chicken agent, cars, and trails
    obs = jax.ops.index_update(obs, jax.ops.index[state["pos"], 4, 0], 1)
    for car_id in range(8):
        car = state["cars"][car_id]
        obs = jax.ops.index_update(obs, jax.ops.index[car[1], car[0], 1], 1)
        # Boundary conditions for cars
        back_x = ((car[3] > 0) * (car[0] - 1) +
                  (1 - (car[3] > 0)) * (car[0] + 1))
        left_out = (back_x < 0)
        right_out = (back_x > 9)
        back_x = left_out * 9 + (1 - left_out) * back_x
        back_x = right_out * 0 + (1 - right_out) * back_x
        # Set trail to be on
        trail_channel = (2 * (jnp.abs(car[3]) == 1) +
                         3 * (jnp.abs(car[3]) == 2) +
                         4 * (jnp.abs(car[3]) == 3) +
                         5 * (jnp.abs(car[3]) == 4) +
                         6 * (jnp.abs(car[3]) == 5))
        obs = jax.ops.index_update(obs, jax.ops.index[car[1], back_x,
                                                      trail_channel], 1)
    return obs


def randomize_cars(rng_input, old_cars, initialize=False):
    """ Randomize car speeds & directions. Reset position if initialize. """
    rng_speed, rng_dirs = jax.random.split(rng_input)
    speeds = jax.random.randint(rng_speed, shape=(8,), minval=1, maxval=6)
    directions = jax.random.choice(rng_dirs, jnp.array([-1, 1]), shape=(8,))
    speeds *= directions
    new_cars = jnp.zeros((8, 4))

    # Loop over all 8 cars and set their data
    for i in range(8):
        # Reset both speeds, directions and positions
        new_cars = jax.ops.index_update(new_cars, jax.ops.index[i, :],
                                    [0, i+1, jnp.abs(speeds[i]), speeds[i]])
        # Reset only speeds and directions
        old_cars = jax.ops.index_update(old_cars, jax.ops.index[i, 2:4],
                                        [jnp.abs(speeds[i]), speeds[i]])

    # Mask the car array manipulation according to initialize
    cars = initialize * new_cars + (1 - initialize) * old_cars
    return cars


reset_freeway = jit(reset)
step_freeway = jit(step)

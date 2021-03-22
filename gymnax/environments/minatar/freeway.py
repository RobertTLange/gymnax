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
    obs = jnp.zeros((10, 10, 6), dtype=bool)
    # Set the position of the chicken agent, cars, and trails
    obs = jax.ops.index_update(obs, jax.ops.index[state["pos"], 4, 0], 1)
    for car_id in range(8):
        car = state["cars"][car_id]
        obs = jax.ops.index_update(obs, jax.ops.index[state["pos"],
                                                      car[1], car[0], 1], 1)
        # Boundary conditions for cars
        back_x = ((car[3] > 0) * (car[0] - 1) +
                  (1 - (car[3] > 0)) * (car[0] + 1))
        left_out = (back_x < 0)
        right_out = (back_x > 9)
        back_x = left_out * 9 + (1 - left_out) * back_x
        back_x = right_out * 0 + (1 - right_out) * back_x
        # TODO: Continue working on state to obs!
        # if(abs(car[3])==1):
        #     trail = self.channels['speed1']
        # elif(abs(car[3])==2):
        #     trail = self.channels['speed2']
        # elif(abs(car[3])==3):
        #     trail = self.channels['speed3']
        # elif(abs(car[3])==4):
        #     trail = self.channels['speed4']
        # elif(abs(car[3])==5):
        #     trail = self.channels['speed5']
        # state[car[1],back_x, trail] = 1
    return obs


reset_freeway = jit(reset)
step_freeway = jit(step)

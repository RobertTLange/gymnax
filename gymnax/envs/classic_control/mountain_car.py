import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible  version of MountainCar-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

# Default environment parameters
params_mountain_car = {"min_position": -1.2,
                       "max_position": 0.6,
                       "max_speed": 0.07,
                       "goal_position": 0.5,
                       "goal_velocity": 0.0,
                       "force": 0.001,
                       "gravity": 0.0025,
                       "max_steps_in_episode": 200}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    position, velocity, done = state
    velocity = (velocity + (action - 1) * params["force"]
                - jnp.cos(3 * position) * params["gravity"])
    velocity = jnp.clip(velocity, -params["max_speed"], params["max_speed"])
    position += velocity
    position = jnp.clip(position, params["min_position"], params["max_position"])
    velocity = velocity * (1 - (position == params["min_position"])
                           * (velocity < 0))
    done = ((position >= params["goal_position"])
            * (velocity >= params["goal_velocity"]))
    reward = -1.0
    state = jnp.array([position, velocity, done])
    return get_obs(state), state, reward, done, {}


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    state = jax.random.uniform(rng_input, shape=(1,),
                               minval=-0.6, maxval=-0.4)
    state = jnp.hstack([state, 0, 0])
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    return jnp.array([state[0], state[1]])


reset_mountain_car = jit(reset)
step_mountain_car = jit(step)

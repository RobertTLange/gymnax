import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py

# Default environment parameters
params_continuous_mountain_car = {"min_action": -1.0,
                                  "max_action": 1,
                                  "min_position": -1.2,
                                  "max_position": 0.6,
                                  "max_speed": 0.07,
                                  "goal_position": 0.45,
                                  "goal_velocity": 0.0,
                                  "power": 0.0015,
                                  "gravity": 0.0025,
                                  "max_steps_in_episode": 999}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    position, velocity, done, timestep = state
    force = jnp.clip(action, params["min_action"], params["max_action"])
    velocity = (velocity + force * params["power"]
                - jnp.cos(3 * position) * params["gravity"])
    velocity = jnp.clip(velocity, -params["max_speed"], params["max_speed"])
    position += velocity
    position = jnp.clip(position, params["min_position"], params["max_position"])
    velocity = velocity * (1 - (position >= params["goal_position"])
                           * (velocity < 0))
    done1 = ((position >= params["goal_position"])
             * (velocity >= params["goal_velocity"]))
    # Check number of steps in episode termination condition
    done_steps = (timestep + 1 > params["max_steps_in_episode"])
    done = jnp.logical_or(done1, done_steps)
    reward = -0.1*action[0]**2 + 100*done1
    state = jnp.hstack([position, velocity, done, timestep+1])
    return get_obs(state), state, reward, done, {}


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    state = jax.random.uniform(rng_input, shape=(1,),
                               minval=-0.6, maxval=-0.4)
    timestep = 0
    state = jnp.hstack([state, 0, 0, timestep])
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    return jnp.array([state[0], state[1]]).squeeze()


reset_continuous_mountain_car = jit(reset)
step_continuous_mountain_car = jit(step)

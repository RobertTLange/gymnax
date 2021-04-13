import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict

# JAX Compatible  version of MountainCar-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

# Default environment parameters
params_mountain_car = FrozenDict({"min_position": -1.2,
                                  "max_position": 0.6,
                                  "max_speed": 0.07,
                                  "goal_position": 0.5,
                                  "goal_velocity": 0.0,
                                  "force": 0.001,
                                  "gravity": 0.0025,
                                  "max_steps_in_episode": 200})


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    velocity = (state["velocity"] + (action - 1) * params["force"]
                - jnp.cos(3 * state["position"]) * params["gravity"])
    velocity = jnp.clip(velocity, -params["max_speed"], params["max_speed"])
    position = state["position"] + velocity
    position = jnp.clip(position, params["min_position"],
                        params["max_position"])
    velocity = velocity * (1 - (position == params["min_position"])
                           * (velocity < 0))
    done1 = ((position >= params["goal_position"])
             * (velocity >= params["goal_velocity"]))
    # Check number of steps in episode termination condition
    done_steps = (state["time"] + 1 > params["max_steps_in_episode"])
    done = jnp.logical_or(done1, done_steps)
    reward = -1.0
    state = {"position": position,
             "velocity": velocity,
             "time": state["time"] + 1,
             "terminal": done}
    return get_obs(state), state, reward, done, {}


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    init_state = jax.random.uniform(rng_input, shape=(),
                                    minval=-0.6, maxval=-0.4)
    state = {"position": init_state,
             "velocity": 0,
             "time": 0,
             "terminal": 0}
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    return jnp.array([state["position"], state["velocity"]])


reset_mountain_car = jit(reset, static_argnums=(1,))
step_mountain_car = jit(step, static_argnums=(1,))

import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict

# JAX Compatible version of Pendulum-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

# Default environment parameters for Pendulum-v0
params_pendulum = FrozenDict({"max_speed": 8,
                              "max_torque": 2.,
                              "dt": 0.05,
                              "g": 10.0,
                              "m": 1.,
                              "l": 1.,
                              "max_steps_in_episode": 200})


def step(rng_input, params, state, u):
    """ Integrate pendulum ODE and return transition. """
    u = jnp.clip(u, -params["max_torque"], params["max_torque"])
    costs = (angle_normalize(state["theta"]) ** 2
             + .1 * state["theta_dot"] ** 2 + .001 * (u ** 2))

    newthdot = state["theta_dot"] + (-3 * params["g"] /
                        (2 * params["l"]) * jnp.sin(state["theta"]
                         + jnp.pi) + 3. /
                        (params["m"] * params["l"] ** 2) * u) * params["dt"]
    newth = state["theta"] + newthdot * params["dt"]
    newthdot = jnp.clip(newthdot, -params["max_speed"], params["max_speed"])
    # Check number of steps in episode termination condition
    done_steps = (state["time"] + 1 > params["max_steps_in_episode"])
    state = {"theta": newth.squeeze(),
             "theta_dot": newthdot.squeeze(),
             "time": state["time"] + 1,
             "terminal": done_steps}
    return get_obs(state), state, -costs[0].squeeze(), done_steps, {}


def reset(rng_input, params):
    """ Reset environment state by sampling theta, theta_dot. """
    high = jnp.array([jnp.pi, 1])
    state = jax.random.uniform(rng_input, shape=(2,),
                               minval=-high, maxval=high)
    timestep = 0
    state = {"theta": state[0],
             "theta_dot": state[1],
             "time": timestep,
             "terminal": 0}
    return get_obs(state), state


def get_obs(state):
    """ Return angle in polar coordinates and change. """
    return jnp.array([jnp.cos(state["theta"]),
                      jnp.sin(state["theta"]),
                      state["theta_dot"]]).squeeze()


def angle_normalize(x):
    """ Normalize the angle - radians. """
    return (((x+jnp.pi) % (2*jnp.pi)) - jnp.pi)


reset_pendulum = jit(reset, static_argnums=(1,))
step_pendulum = jit(step, static_argnums=(1,))

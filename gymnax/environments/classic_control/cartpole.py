import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict

# JAX Compatible version of CartPole-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Default environment parameters for CartPole-v0
params_cartpole = FrozenDict({"gravity": 9.8,
                              "masscart": 1.0,
                              "masspole": 0.1,
                              "total_mass": 1.0 + 0.1,  # (masscart + masspole)
                              "length": 0.5,
                              "polemass_length": 0.05,  # (masspole * length)
                              "force_mag": 10.0,
                              "tau": 0.02,
                              "theta_threshold_radians": 12*2*jnp.pi/360,
                              "x_threshold": 2.4,
                              "max_steps_in_episode": 200})


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    force = params["force_mag"] * action -params["force_mag"]*(1-action)
    costheta = jnp.cos(state["theta"])
    sintheta = jnp.sin(state["theta"])

    temp = (force + params["polemass_length"] * state["theta_dot"] ** 2
            * sintheta) / params["total_mass"]
    thetaacc = ((params["gravity"] * sintheta - costheta * temp) /
                (params["length"] * (4.0 / 3.0 - params["masspole"]
                 * costheta ** 2 / params["total_mass"])))
    xacc = (temp - params["polemass_length"] * thetaacc * costheta
            / params["total_mass"])

    # Only default Euler integration option available here!
    x = state["x"] + params["tau"] * state["x_dot"]
    x_dot = state["x_dot"] + params["tau"] * xacc
    theta = state["theta"] + params["tau"] * state["theta_dot"]
    theta_dot = state["theta_dot"] + params["tau"] * thetaacc

    # Check termination criteria
    done1 = jnp.logical_or(x < -params["x_threshold"],
                           x > params["x_threshold"])
    done2 = jnp.logical_or(theta < -params["theta_threshold_radians"],
                           theta > params["theta_threshold_radians"])

    # Important: Reward is based on termination is previous step transition
    reward = 1.0 - state["terminal"]

    # Check number of steps in episode termination condition
    done_steps = (state["time"] + 1 > params["max_steps_in_episode"])
    done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
    state = {"x": x,
             "x_dot": x_dot,
             "theta": theta,
             "theta_dot": theta_dot,
             "time": state["time"] + 1,
             "terminal": done}
    return get_obs(state), state, reward, done, {}


def get_obs(state):
    """ Return observation from raw state trafo. """
    return jnp.array([state["x"], state["x_dot"],
                      state["theta"], state["theta_dot"]])


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    init_state = jax.random.uniform(rng_input, minval=-0.05,
                                    maxval=0.05, shape=(4,))
    timestep = 0
    state = {"x": init_state[0],
             "x_dot": init_state[1],
             "theta": init_state[2],
             "theta_dot": init_state[3],
             "time": 0,
             "terminal": 0}
    return get_obs(state), state


reset_cartpole = jit(reset, static_argnums=(1,))
step_cartpole = jit(step, static_argnums=(1,))

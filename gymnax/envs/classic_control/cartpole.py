import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of CartPole-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Default environment parameters for CartPole-v0
params_cartpole = {"gravity": 9.8,
                   "masscart": 1.0,
                   "masspole": 0.1,
                   "total_mass": 1.0 + 0.1,  # (masscart + masspole)
                   "length": 0.5,
                   "polemass_length": 0.05,  # (masspole * length)
                   "force_mag": 10.0,
                   "tau": 0.02,
                   "theta_threshold_radians": 12*2*jnp.pi/360,
                   "x_threshold": 2.4}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    x, x_dot, theta, theta_dot, just_done = state
    force = params["force_mag"] * action -params["force_mag"]*(1-action)
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    temp = (force + params["polemass_length"] * theta_dot ** 2
            * sintheta) / params["total_mass"]
    thetaacc = ((params["gravity"] * sintheta - costheta * temp) /
                (params["length"] * (4.0 / 3.0 - params["masspole"]
                 * costheta ** 2 / params["total_mass"])))
    xacc = (temp - params["polemass_length"] * thetaacc * costheta
            / params["total_mass"])

    # Only default Euler integration option available here!
    x = x + params["tau"] * x_dot
    x_dot = x_dot + params["tau"] * xacc
    theta = theta + params["tau"] * theta_dot
    theta_dot = theta_dot + params["tau"] * thetaacc

    # Check termination criteria
    done1 = jnp.logical_or(x < -params["x_threshold"],
                           x > params["x_threshold"])
    done2 = jnp.logical_or(theta < -params["theta_threshold_radians"],
                           theta > params["theta_threshold_radians"])
    done = jnp.logical_or(done1, done2)
    state = jnp.hstack([x, x_dot, theta, theta_dot, done])
    reward = 1.0 - just_done
    return get_obs(state), state, reward, done, {}


def get_obs(state):
    """ Return observation from raw state trafo. """
    return jnp.array([state[0], state[1], state[2], state[3]])


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    state = jax.random.uniform(rng_input, minval=-0.05, maxval=0.05, shape=(4,))
    state = jnp.hstack([state, 0])
    return get_obs(state), state


reset_cartpole = jit(reset)
step_cartpole = jit(step)


# Angle limit set to 2 * theta_threshold_radians so failing observation
# is still within bounds.
#high = np.array([self.x_threshold * 2,
#                 np.finfo(np.float32).max,
#                 self.theta_threshold_radians * 2,
#                 np.finfo(np.float32).max],
#                dtype=np.float32)
#self.observation_space = spaces.Box(-high, high, dtype=np.float32)

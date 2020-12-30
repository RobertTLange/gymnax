import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Acrobot-v1 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
# NOTES: We only implement the default 'book' version

# Default environment parameters
params_acrobot = {"dt": 0.2,
                  "link_length_1": 1.0,
                  "link_length_2": 1.0,
                  "link_mass_1": 1.0,
                  "link_mass_2": 1.0,
                  "link_com_pos_1": 0.5,
                  "link_com_pos_2": 0.5,
                  "link_moi": 1.0,
                  "max_vel_1": 4*jnp.pi,
                  "max_vel_2": 9*jnp.pi,
                  "available_torque": jnp.array([-1., 0., +1]),
                  "torque_noise_max": 0.0}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    torque = params["available_torque"][action]
    # Add noise to force action - always sample - conditionals in JAX
    torque = torque + jax.random.uniform(rng_input, shape=(1,),
                                         minval=-params["torque_noise_max"], maxval=params["torque_noise_max"])

    # Augment state with force action so it can be passed to ds/dt
    s_augmented = jnp.append(state, torque)
    ns = rk4(s_augmented, params)
    joint_angle1 = wrap(ns[0], -jnp.pi, jnp.pi)
    joint_angle2 = wrap(ns[1], -jnp.pi, jnp.pi)
    vel1 = jnp.clip(ns[2], -params["max_vel_1"], params["max_vel_1"])
    vel2 = jnp.clip(ns[3], -params["max_vel_2"], params["max_vel_2"])
    state = jnp.array([joint_angle1, joint_angle2, vel1, vel2])
    done = (-jnp.cos(state[0]) - jnp.cos(state[1] + state[0]) > 1.)
    reward = -1. * (1-done)
    return get_obs(state), state, reward, done, {}


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    state = jax.random.uniform(rng_input, shape=(4,), minval=-0.1, maxval=0.1)
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]),
                     jnp.cos(state[1]), jnp.sin(state[1]),
                     state[2], state[3]])
    return obs


reset_acrobot = jit(reset)
step_acrobot = jit(step)


def dsdt(s_augmented, t, params):
    """ Compute time derivative of the state change - Use for ODE int. """
    m1, m2 = params["link_mass_1"], params["link_mass_2"]
    l1 = params["link_length_1"]
    lc1, lc2 = params["link_com_pos_1"], params["link_com_pos_2"]
    I1, I2 = params["link_moi"], params["link_moi"]
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1, theta2, dtheta1, dtheta2 = s
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2) + phi2
    ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.])


def wrap(x, m, M):
    """ For example, m = -180, M = 180 (degrees), x = 360 --> returns 0. """
    diff = M - m
    diff_x_M = x - M
    diff_x_m = x - m
    go_down = diff_x_M > 0
    go_up = diff_x_m < 0

    how_often = (jnp.ceil(diff_x_M/diff)*go_down
                 + jnp.ceil(diff_x_m/diff)*go_up)
    #while x > M:
    #    x = x - diff
    #while x < m:
    #    x = x + diff
    x_out = x - how_often*diff*go_down + how_often*diff*go_up
    return x_out


def rk4(y0, params):
    """ Runge-Kutta integration of ODE - Difference to OpenAI: Only 1 step! """
    dt2 = params["dt"] / 2.0
    k1 = dsdt(y0, 0, params)
    k2 = dsdt(y0 + dt2 * k1, dt2, params)
    k3 = dsdt(y0 + dt2 * k2, dt2, params)
    k4 = dsdt(y0 + params["dt"] * k3, params["dt"], params)
    yout = y0 + params["dt"] / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout

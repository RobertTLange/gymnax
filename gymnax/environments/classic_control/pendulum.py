import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Pendulum-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

# Default environment parameters for Pendulum-v0
params_pendulum = {"max_speed": 8,
                   "max_torque": 2.,
                   "dt": 0.05,
                   "g": 10.0,
                   "m": 1.,
                   "l": 1.,
                   "max_steps_in_episode": 200}


def step(rng_input, params, state, u):
    """ Integrate pendulum ODE and return transition. """
    th, thdot, timestep = state
    u = jnp.clip(u, -params["max_torque"], params["max_torque"])
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

    newthdot = thdot + (-3 * params["g"] /
                        (2 * params["l"]) * jnp.sin(th + jnp.pi) + 3. /
                        (params["m"] * params["l"] ** 2) * u) * params["dt"]
    newth = th + newthdot * params["dt"]
    newthdot = jnp.clip(newthdot, -params["max_speed"], params["max_speed"])
    # Check number of steps in episode termination condition
    done_steps = (timestep + 1 > params["max_steps_in_episode"])
    state = jnp.hstack([newth, newthdot, timestep+1])
    return get_obs(state), state.squeeze(), -costs[0].squeeze(), done_steps, {}


def reset(rng_input, params):
    """ Reset environment state by sampling theta, thetadot. """
    high = jnp.array([jnp.pi, 1])
    state = jax.random.uniform(rng_input, shape=(2,),
                               minval=-high, maxval=high)
    timestep = 0
    state = jnp.hstack([state, timestep])
    return get_obs(state), state


def get_obs(state):
    """ Return angle in polar coordinates and change. """
    th, thdot = state[0], state[1]
    return jnp.array([jnp.cos(th), jnp.sin(th), thdot]).squeeze()


def angle_normalize(x):
    """ Normalize the angle - radians. """
    return (((x+jnp.pi) % (2*jnp.pi)) - jnp.pi)


reset_pendulum = jit(reset)
step_pendulum = jit(step)

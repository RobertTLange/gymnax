import jax
import jax.numpy as jnp
from jax import jit

"""
Template for JAX Compatible Environments.
-------------------------------------------
Steps for adding your own environments/contributing to the repository:
0. Fork and clone the repository.
1. Modify step, reset, get_obs functions from template for your needs.
2. Add the environment id/name to registration.py and the __init__.py imports.
3. Add a unit test to the tests directory and check correctness.
4. Add any reference to the original version/documentation of the environment.
5. [OPTIONAL] Add a link to the Readme of projects that use this environment.
6. [OPTIONAL] Add an example notebook with a step transition/episode rollout.
7. [OPTIONAL] Run a speed benchmark comparing performance on different devices.
8. Open a pull request.
"""

# Default environment parameters
params_env_name = {"max_speed": 8,}


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    action = jnp.clip(action,
                      -params["max_speed"],
                      params["max_speed"])
    state = 0
    reward = 0
    done = False
    info = {}
    return get_obs(state), state, reward, done, info


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    high = jnp.array([jnp.pi, 1])
    state = jax.random.uniform(rng_input, shape=(2,),
                               minval=-high, maxval=high)
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    obs = 2*state
    return obs


reset_env_name = jit(reset)
step_env_name = jit(step)

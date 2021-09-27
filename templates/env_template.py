import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey

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


class YourCoolEnv(environment.Environment):
    def __init__(self):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"max_steps_in_episode": 100})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        action = jnp.clip(action, -1, 1)
        reward = 0

        # Check game condition & no. steps for termination condition
        state["time"] += 1
        done = self.is_terminal(state)
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done, info)

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        high = jnp.array([jnp.pi, 1])
        state = {}
        state["var"] = jax.random.uniform(key, shape=(2,),
                                          minval=-high, maxval=high)
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        return obs

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        return False

    @property
    def name(self) -> str:
        """ Environment name. """
        return "YourCoolEnv-v0"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(2)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(-1, 1, (1,))

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})

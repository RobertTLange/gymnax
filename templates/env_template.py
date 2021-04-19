import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

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
        self.env_params = {}

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        action = jnp.clip(action, -1, 1)
        reward = 0
        done = False
        info = {}
        return get_obs(state), state, reward, done, info

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        high = jnp.array([jnp.pi, 1])
        state = {}
        state["var"] = jax.random.uniform(key, shape=(2,),
                                          minval=-high, maxval=high)
        return get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        return state

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
        return spaces.Dict(["var"])

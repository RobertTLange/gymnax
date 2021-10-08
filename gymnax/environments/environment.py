from jax import lax

from gymnax.utils import jittable
from gymnax.utils.frozen_dict import unfreeze, FrozenDict

import abc
from typing import Tuple, Union
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class Environment(jittable.Jittable, metaclass=abc.ABCMeta):
    """Jittable abstract base class for all gymnax Environments."""

    @abc.abstractmethod
    def step(
        self, key: PRNGKey, state: dict, action: Union[int, float]
    ) -> Tuple[Array, dict, float, bool]:
        """Performs step transitions in the environment."""

    @abc.abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """Performs resetting of environment."""

    @abc.abstractmethod
    def get_obs(self, state: dict) -> Array:
        """Applies observation function to state."""

    @abc.abstractmethod
    def is_terminal(self, state: dict) -> bool:
        """Check whether state transition is terminal."""

    def discount(self, state: dict, params: dict) -> float:
        """Return a discount of zero if the episode has terminated."""
        return lax.select(self.is_terminal(state, params), 0.0, 1.0)

    def update_env_params(
        self, p_name: str, p_value: Union[str, float, int, bool, Array]
    ):
        "Update single environment parameter - Unfreeze & freeze dictionary."
        env_dict = unfreeze(self.env_params)
        env_dict[p_name] = p_value
        self.env_params = FrozenDict(env_dict)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def params(self):
        """Returns environment params."""
        return self.env_params

    @property
    @abc.abstractmethod
    def action_space(self):
        """Action space of the environment."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """Observation space of the environment."""

    @property
    @abc.abstractmethod
    def state_space(self):
        """State space of the environment."""

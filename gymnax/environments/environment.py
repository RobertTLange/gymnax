import abc
from typing import Sequence, Tuple, Union
import chex
from gymnax.utils import jittable
from gymnax.utils.frozen_dict import freeze, unfreeze, FrozenDict
import jax
import jax.numpy as jnp

Array = chex.Array
PRNGKey = chex.PRNGKey


class Environment(jittable.Jittable, metaclass=abc.ABCMeta):
    """Jittable abstract base class for all gymnax Environments."""

    @abc.abstractmethod
    def step(self, key: PRNGKey, state: dict, action: Union[int, float]
             ) -> Tuple[Array, dict,float, bool, dict]:
        """ Performs step transitions in the environment."""

    @abc.abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Performs resetting of environment."""

    @abc.abstractmethod
    def get_obs(self, state: dict) -> Array:
        """ Applies observation function to state."""

    def update_env_params(self, p_name: str,
                          p_value: Union[str, float, int, bool, Array]):
        " Update single environment parameter - Unfreeze & freeze dictionary. "
        env_dict = unfreeze(self.env_params)
        env_dict["p_name"] = p_value
        self.env_params = FrozenDict(env_dict)

    @property
    def name(self) -> str:
        """Distribution name."""
        return type(self).__name__

    @property
    @abc.abstractmethod
    def action_space(self):
        """ Action space of the environment."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """ Action space of the environment."""

    @property
    @abc.abstractmethod
    def state_space(self):
        """ Action space of the environment."""

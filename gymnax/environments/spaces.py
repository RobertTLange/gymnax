from typing import Sequence, Tuple, Union
import chex
from gymnax.utils import jittable
import jax

Array = chex.Array
PRNGKey = chex.PRNGKey


class Discrete(jittable.Jittable):
    """ Jittable class for discrete gymnax spaces.
        TODO: For now this is a 1d space. Make composable for multi-dim spaces.
    """
    def __init__(self, num_categories: int):
        self.num_categories = num_categories

    def sample(self, rng: PRNGKey):
        """ Sample random action uniformly from set of categorical choices. """
        return jax.random.randint(rng, shape=(), minval=0,
                                  maxval=self.num_categories-1)


class Continuous(jittable.Jittable):
    """ Jittable class for continuous gymnax spaces.
        TODO: For now this is a 1d space. Make composable for multi-dim spaces.
    """
    def __init__(self, minval: float, maxval: float):
        self.minval = minval
        self.maxval = maxval

    def sample(self, rng: PRNGKey):
        """ Sample random action uniformly from 1D continuous range. """
        return jax.random.uniform(rng, shape=(), minval=minval,
                                  maxval=maxval)


class Box():
    """ Jittable class for array-shaped gymnax spaces.
    """
    def __init__(self, low: float, high: float, shape: tuple):
        self.low = low
        self.high = high
        self.shape = shape


class Dict():
    """ Class for dictionary of simpler jittable spaces. """
    def __init__(self, dkeys: dict):
        self.dkeys = dkeys

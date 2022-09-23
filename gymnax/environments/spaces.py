from typing import Tuple, Sequence, Any, Dict
from collections import OrderedDict
import chex
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces as gspc


class Space:
    """
    Minimal jittable class for abstract gymnax space.
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: jnp.int_) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """
    Minimal jittable class for discrete gymnax spaces.
    TODO: For now this is a 1d space. Make composable for multi-discrete.
    """

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """
    Minimal jittable class for array-shaped gymnax spaces.
    TODO: Add unboundedness - sampling from other distributions, etc.
    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond


class Dict(Space):
    """Minimal jittable class for dictionary of simpler jittable spaces."""

    def __init__(self, spaces: Dict[Any, Space]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: chex.PRNGKey) -> Dict:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return OrderedDict(
            [
                (k, self.spaces[k].sample(key_split[i]))
                for i, k in enumerate(self.spaces)
            ]
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, Dict)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for k, space in self.spaces.items():
            out_of_space += 1 - space.contains(getattr(x, k))
        return out_of_space == 0


class Tuple(Space):
    """Minimal jittable class for tuple (product) of jittable spaces."""

    def __init__(self, spaces: Sequence[Space]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: chex.PRNGKey) -> Tuple[chex.Array]:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return tuple(
            [self.spaces[k].sample(key_split[i]) for i, k in enumerate(self.spaces)]
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, tuple)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for space in self.spaces:
            out_of_space += 1 - space.contains(x)
        return out_of_space == 0


def gymnax_space_to_gym_space(space: Space) -> gspc.Space:
    """Convert Gymnax space to equivalent Gym space"""
    if isinstance(space, Discrete):
        return gspc.Discrete(space.n)
    elif isinstance(space, Box):
        low = float(space.low) if (np.isscalar(space.low) or space.low.size == 1) else np.array(space.low)
        high = float(space.high) if (np.isscalar(space.high) or space.low.size == 1) else np.array(space.high)
        return gspc.Box(low, high, space.shape, space.dtype)
    elif isinstance(space, Dict):
        return gspc.Dict({k: gymnax_space_to_gym_space(v) for k, v in space.spaces})
    elif isinstance(space, Tuple):
        return gspc.Tuple(space.spaces)
    else:
        raise NotImplementedError(
            f"Conversion of {space.__class__.__name__} not supported"
        )

"""Gymnax space classes."""

import collections
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces as gspc


class Space:
    """Minimal jittable class for abstract gymnax space."""

    def sample(self, key: jax.Array) -> jax.Array:
        raise NotImplementedError

    def contains(self, x: jax.Array) -> Any:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete Gymnax spaces.

    Args:
        num_categories: Number of integer values in the space.
        dtype: Keyword-only action dtype. Supports ``int32`` by default and
            ``int64`` when JAX x64 mode is enabled.
    """

    def __init__(self, num_categories: int, *, dtype: Any = jnp.int32):
        assert num_categories >= 0
        try:
            dtype = jnp.dtype(dtype)
        except TypeError as error:
            raise ValueError("Discrete dtype must be int32 or int64") from error
        if dtype == jnp.dtype(jnp.int64) and not jax.config.read("jax_enable_x64"):
            raise ValueError("Discrete(dtype=int64) requires JAX x64 mode")
        if dtype not in (jnp.dtype(jnp.int32), jnp.dtype(jnp.int64)):
            raise ValueError("Discrete dtype must be int32 or int64")

        self.n = num_categories
        self.shape = ()
        self.dtype = dtype

    def __repr__(self) -> str:
        """Return a concise representation for interactive inspection."""
        return f"Discrete({self.n}, dtype={self.dtype.name})"

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            key,
            shape=self.shape,
            minval=0,
            maxval=self.n,
            dtype=self.dtype,
        )

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        x = jnp.asarray(x, dtype=self.dtype)
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(
            x >= jnp.asarray(0, dtype=self.dtype),
            x < jnp.asarray(self.n, dtype=self.dtype),
        )
        return range_cond


class Box(Space):
    """Minimal jittable class for array-shaped gymnax spaces."""

    def __init__(
        self,
        low: jnp.ndarray | float,
        high: jnp.ndarray | float,
        shape: Any,  # Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        """Return the configured bounds, shape, and dtype."""
        return (
            f"Box(low={_format_bound(self.low)}, high={_format_bound(self.high)}, "
            f"shape={self.shape}, dtype={np.dtype(self.dtype).name})"
        )

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            key, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond


class Dict(Space):
    """Minimal jittable class for dictionary of simpler jittable spaces."""

    def __init__(self, spaces: Any):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def __repr__(self) -> str:
        """Return the representations of the keyed child spaces."""
        return f"Dict({self.spaces!r})"

    def sample(self, key: jax.Array) -> Any:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(key, self.num_spaces)
        return collections.OrderedDict(
            [
                (k, self.spaces[k].sample(key_split[i]))
                for i, k in enumerate(self.spaces)
            ]
        )

    def contains(self, x: jax.Array) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, Dict)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for k, space in self.spaces.items():
            out_of_space += 1 - space.contains(getattr(x, k)).astype(jnp.int32)
        return out_of_space == 0


class Tuple(Space):
    """Minimal jittable class for tuple (product) of jittable spaces."""

    def __init__(self, spaces: Sequence[Space]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def __repr__(self) -> str:
        """Return the representations of the child spaces."""
        return f"Tuple({tuple(self.spaces)!r})"

    def sample(self, key: jax.Array) -> Any:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(key, self.num_spaces)
        return tuple([s.sample(key_split[i]) for i, s in enumerate(self.spaces)])

    def contains(self, x: jax.Array) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, tuple)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for i, space in enumerate(self.spaces):
            out_of_space += 1 - space.contains(x[i])
        return out_of_space == 0


def gymnax_space_to_gym_space(space: Space) -> gspc.Space:
    """Convert Gymnax space to equivalent Gym space."""
    if isinstance(space, Discrete):
        return gspc.Discrete(space.n)
    elif isinstance(space, Box):
        low = (
            float(space.low)
            if (np.isscalar(space.low) or space.low.size == 1)
            else np.array(space.low)
        )
        high = (
            float(space.high)
            if (np.isscalar(space.high) or space.low.size == 1)
            else np.array(space.high)
        )
        return gspc.Box(low, high, space.shape, space.dtype)
    elif isinstance(space, Dict):
        return gspc.Dict(
            {
                key: gymnax_space_to_gym_space(value)
                for key, value in space.spaces.items()
            }
        )
    elif isinstance(space, Tuple):
        children = tuple(gymnax_space_to_gym_space(child) for child in space.spaces)
        return gspc.Tuple(children)
    else:
        raise NotImplementedError(
            f"Conversion of {space.__class__.__name__} not supported"
        )


def _format_bound(value: jax.Array | float) -> str:
    """Format scalar and array bounds without JAX device details."""
    array = np.asarray(value)
    if array.ndim == 0:
        return repr(array.item())
    return np.array2string(array, threshold=6)

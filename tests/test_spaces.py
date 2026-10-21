"""Tests for Gymnax space contracts."""

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gymnax.environments import spaces
from gymnax.environments.classic_control.cartpole import CartPole
from gymnax.environments.spaces import Discrete


def test_discrete_defaults_to_int32_for_sampling_and_contains():
    """Discrete spaces keep the historic int32 action contract by default."""
    space = Discrete(3)

    sample = space.sample(jax.random.key(0))

    assert space.dtype == jnp.dtype(jnp.int32)
    assert sample.dtype == jnp.dtype(jnp.int32)
    assert bool(space.contains(sample))


def test_discrete_normalizes_int32_dtype():
    """Equivalent int32 dtype specifications produce the supported dtype."""
    space = Discrete(3, dtype=np.dtype("int32"))

    assert space.dtype == jnp.dtype(jnp.int32)
    assert space.sample(jax.random.key(0)).dtype == jnp.dtype(jnp.int32)


def test_discrete_dtype_is_keyword_only():
    """The dtype extension cannot alter the positional constructor API."""
    with pytest.raises(TypeError):
        Discrete(3, jnp.int32)


@pytest.mark.skipif(
    not jax.config.x64_enabled,
    reason="int64 Discrete spaces require JAX x64 mode",
)
def test_discrete_uses_requested_int64_dtype_when_x64_is_enabled():
    """Sampling and membership checks use the configured int64 dtype."""
    space = Discrete(3, dtype=jnp.int64)
    sample = space.sample(jax.random.key(0))

    assert space.dtype == jnp.dtype(jnp.int64)
    assert sample.dtype == jnp.dtype(jnp.int64)
    assert bool(space.contains(sample))
    assert not bool(space.contains(jnp.array(3, dtype=jnp.int64)))


@pytest.mark.skipif(
    jax.config.x64_enabled,
    reason="int64 is supported when JAX x64 mode is enabled",
)
def test_discrete_rejects_int64_when_x64_is_disabled():
    """The constructor does not silently truncate the requested int64 dtype."""
    with pytest.raises(ValueError, match="requires JAX x64 mode"):
        Discrete(3, dtype=jnp.int64)


@pytest.mark.parametrize(
    "dtype", [jnp.int8, jnp.uint32, jnp.bool_, jnp.float32, "not-a-dtype"]
)
def test_discrete_rejects_unsupported_dtypes(dtype):
    """Only int32 and x64-enabled int64 are valid Discrete dtypes."""
    with pytest.raises(ValueError, match="must be int32 or int64"):
        Discrete(3, dtype=dtype)


def test_tuple_space_converts_recursively_for_gymnasium_adapters():
    """Tuple leaves become native Gymnasium spaces, not Gymnax space objects."""
    converted = spaces.gymnax_space_to_gym_space(
        spaces.Tuple((spaces.Discrete(2), spaces.Box(-1.0, 1.0, (2,))))
    )

    assert converted.contains((0, np.zeros((2,), dtype=np.float32)))


def test_dict_space_converts_recursively_for_gymnasium_adapters():
    """Dictionary leaves become native Gymnasium spaces by their keys."""
    converted = spaces.gymnax_space_to_gym_space(
        spaces.Dict(
            {
                "choice": spaces.Discrete(2),
                "vector": spaces.Box(-1.0, 1.0, (2,)),
            }
        )
    )

    assert converted.contains({"choice": 0, "vector": np.zeros((2,), dtype=np.float32)})


def test_spaces_have_readable_constructor_style_representations():
    """Interactive inspection reveals bounds, dtypes, and nested children."""
    discrete = spaces.Discrete(3)
    box = spaces.Box(-1.0, 1.0, (2,))
    composite = spaces.Dict({"choice": discrete, "vector": box})
    tuple_space = spaces.Tuple((discrete, box))

    assert repr(discrete) == "Discrete(3, dtype=int32)"
    assert repr(box) == "Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)"
    assert repr(composite) == (
        "Dict({'choice': Discrete(3, dtype=int32), "
        "'vector': Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)})"
    )
    assert repr(tuple_space) == (
        "Tuple((Discrete(3, dtype=int32), "
        "Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)))"
    )


@pytest.mark.parametrize("mapping_type", [dict, OrderedDict])
def test_dict_contains_accepts_sampled_mappings(mapping_type):
    """Nested dictionary samples must pass their own space's membership check."""
    space = spaces.Dict(
        {
            "choice": spaces.Discrete(3),
            "vector": spaces.Box(-1.0, 1.0, (2,)),
            "nested": spaces.Dict({"flag": spaces.Discrete(2)}),
        }
    )

    def check_sample(key):
        sample = mapping_type(space.sample(key))
        return space.contains(sample)

    keys = jax.random.split(jax.random.key(0), 3)
    assert bool(jnp.all(jax.jit(jax.vmap(check_sample))(keys)))
    invalid = mapping_type(space.sample(keys[0]))
    invalid["choice"] = jnp.array(3)
    assert not bool(jax.jit(space.contains)(invalid))


def test_dict_contains_keeps_attribute_based_environment_states():
    """Built-in dataclass environment states still pass membership checks."""
    env = CartPole()
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)

    assert bool(jax.jit(env.state_space(params).contains)(state))

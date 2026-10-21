"""Regression tests for Gymnasium bound conversion."""

import jax.numpy as jnp
import numpy as np
import pytest

from gymnax.environments import spaces


@pytest.mark.parametrize(
    "low, high",
    [
        (0.0, jnp.array([1.0, 2.0])),
        (jnp.array([0.0]), jnp.array([1.0, 2.0])),
        (jnp.array([0.0, 0.0]), jnp.array([2.0])),
    ],
    ids=["scalar-low", "singleton-low", "singleton-high"],
)
def test_box_bounds_convert_independently(low, high):
    """Broadcastable scalar and array bounds keep their per-coordinate values."""
    box = spaces.Box(low, high, (2,))

    converted = spaces.gymnax_space_to_gym_space(box)

    expected_low = np.broadcast_to(np.asarray(low), (2,))
    expected_high = np.broadcast_to(np.asarray(high), (2,))
    np.testing.assert_array_equal(converted.low, expected_low)
    np.testing.assert_array_equal(converted.high, expected_high)
    midpoint = ((expected_low + expected_high) / 2).astype(np.float32)
    assert converted.contains(midpoint)
    assert not converted.contains((expected_high + 1).astype(np.float32))


def test_box_conversion_does_not_mutate_caller_bounds():
    """Gymnasium's integer-bound normalization must not alter caller arrays."""
    low = np.array([-np.inf, 0.0], dtype=np.float64)
    high = np.array([np.inf, 1.0], dtype=np.float64)
    original_low, original_high = low.copy(), high.copy()

    converted = spaces.gymnax_space_to_gym_space(
        spaces.Box(low, high, (2,), dtype=jnp.int32)
    )

    assert converted.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(low, original_low)
    np.testing.assert_array_equal(high, original_high)

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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bool_])
@pytest.mark.parametrize(
    "low, high",
    [
        (False, True),
        (np.bool_(False), np.bool_(True)),
        (np.array(False), np.array(True)),
        (jnp.array(False), jnp.array(True)),
        (jnp.array([False]), jnp.array([True])),
    ],
    ids=["python", "numpy-scalar", "numpy-array", "jax-array", "jax-singleton"],
)
def test_box_boolean_bounds_preserve_values_and_dtype(low, high, dtype):
    """Boolean bounds remain convertible for floating and boolean Box spaces."""
    converted = spaces.gymnax_space_to_gym_space(
        spaces.Box(low, high, (2,), dtype=dtype)
    )

    assert converted.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(converted.low, np.zeros(2, dtype=dtype))
    np.testing.assert_array_equal(converted.high, np.ones(2, dtype=dtype))
    assert converted.contains(np.array([False, True], dtype=dtype))


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


@pytest.mark.parametrize(
    "low, high",
    [
        (jnp.array([-1.0, -2.0, -3.0]), 4.0),
        (-4.0, jnp.array([1.0, 2.0, 3.0])),
        (jnp.array([[-1.0], [-2.0]]), jnp.array([1.0, 2.0, 3.0])),
        (jnp.array([-1.0, -2.0, -3.0]), jnp.ones((2, 3))),
    ],
    ids=["row-low", "row-high", "column-and-row", "row-and-full"],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_box_conversion_broadcasts_array_bounds(low, high, dtype):
    """Array bounds broadcast to the declared multidimensional Box shape."""
    box = spaces.Box(low, high, (2, 3), dtype=dtype)

    converted = spaces.gymnax_space_to_gym_space(box)

    assert converted.shape == box.shape
    assert converted.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(converted.low, np.broadcast_to(low, box.shape))
    np.testing.assert_array_equal(converted.high, np.broadcast_to(high, box.shape))
    assert converted.contains(np.zeros(box.shape, dtype=dtype))


def test_broadcast_conversion_copies_numpy_bounds():
    """Broadcasting does not share storage with caller-owned bounds."""
    low = np.array([[-np.inf], [0.0]])
    high = np.array([np.inf, 1.0, 2.0])
    before_low, before_high = low.copy(), high.copy()

    converted = spaces.gymnax_space_to_gym_space(
        spaces.Box(low, high, (2, 3), dtype=jnp.int32)
    )

    np.testing.assert_array_equal(low, before_low)
    np.testing.assert_array_equal(high, before_high)
    assert converted.low.shape == converted.high.shape == (2, 3)
    assert converted.low.flags.writeable
    assert converted.high.flags.writeable


def test_broadcast_box_bounds_in_gymnasium_wrappers():
    """Single and vector adapters expose broadcastable custom observation bounds."""
    import jax

    from gymnax.environments.classic_control.cartpole import CartPole
    from gymnax.wrappers.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper

    class MatrixCartPole(CartPole):
        def get_obs(self, state, params=None, key=None):
            return super().get_obs(state, params, key).astype(jnp.float32).reshape(2, 2)

        def observation_space(self, params):
            return spaces.Box(jnp.array([-5.0, -5.0]), 5.0, (2, 2))

    env = MatrixCartPole()
    single = GymnaxToGymWrapper(env)
    vector = GymnaxToVectorGymWrapper(env, num_envs=2)

    observation, _ = single.reset(seed=3)
    assert single.observation_space.contains(np.asarray(observation))
    assert vector.single_observation_space.shape == (2, 2)
    assert vector.observation_space.shape == (2, 2, 2)
    observations, _ = vector.reset(seed=3)
    assert vector.observation_space.contains(np.asarray(observations))
    expected, _ = env.reset(jax.random.split(jax.random.key(3))[1])
    np.testing.assert_array_equal(observation, expected)


def test_box_conversion_keeps_empty_and_boolean_bounds():
    """Empty arrays and broadcastable boolean bounds retain their shape and dtype."""
    empty = spaces.gymnax_space_to_gym_space(
        spaces.Box(np.empty((0,)), np.empty((0,)), (2, 0))
    )
    assert empty.shape == (2, 0)
    boolean = spaces.gymnax_space_to_gym_space(
        spaces.Box(
            np.array([False, True]), np.array([[True], [True]]), (2, 2), jnp.bool_
        )
    )
    np.testing.assert_array_equal(boolean.low, [[False, True], [False, True]])
    np.testing.assert_array_equal(boolean.high, np.ones((2, 2), dtype=bool))

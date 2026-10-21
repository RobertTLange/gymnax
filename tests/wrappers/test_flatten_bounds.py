"""Regression tests for flattened observation bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gymnax.environments import spaces
from gymnax.environments.bsuite.catch import Catch
from gymnax.wrappers import FlattenObservationWrapper


class ArrayBoundsCatch(Catch):
    """Catch with equivalent or looser array-valued observation bounds."""

    def __init__(self, low, high):
        super().__init__(rows=2, columns=3)
        self.low = low
        self.high = high

    def observation_space(self, params):
        return spaces.Box(self.low, self.high, (2, 3), dtype=jnp.float32)


@pytest.mark.parametrize(
    "low, high",
    [
        (-jnp.arange(1.0, 7.0).reshape(2, 3), jnp.arange(1.0, 7.0).reshape(2, 3)),
        (0.0, jnp.array([1.0, 2.0, 3.0])),
    ],
    ids=["per-coordinate", "broadcast-columns"],
)
def test_flatten_wrapper_preserves_array_bounds(low, high):
    """Flattened bounds must contain reset, step and terminal observations."""
    env = ArrayBoundsCatch(low, high)
    params = env.default_params.replace(max_steps_in_episode=1)
    wrapper = FlattenObservationWrapper(env)
    flat_space = wrapper.observation_space(params)
    key = jax.random.key(0)
    observation, state = wrapper.reset(key, params)

    assert flat_space.shape == observation.shape == (6,)
    assert bool(flat_space.contains(observation))
    for actual, original in [(flat_space.low, low), (flat_space.high, high)]:
        np.testing.assert_array_equal(
            np.broadcast_to(np.asarray(actual), (6,)),
            np.broadcast_to(np.asarray(original), (2, 3)).reshape(-1),
        )
    assert bool(flat_space.contains(flat_space.sample(key)))

    observation, _, _, _, truncated, info = wrapper.step(key, state, 1, params)
    assert bool(truncated)
    assert bool(flat_space.contains(observation))
    assert bool(flat_space.contains(info["final_observation"]))


def test_flatten_wrapper_keeps_scalar_bound_representation():
    """Built-in scalar bounds retain their values, rank and declared dtype."""
    env = Catch(rows=2, columns=3)
    original = env.observation_space(env.default_params)
    flattened = FlattenObservationWrapper(env).observation_space(env.default_params)

    assert jnp.ndim(flattened.low) == jnp.ndim(flattened.high) == 0
    assert flattened.low == original.low
    assert flattened.high == original.high
    assert flattened.dtype == original.dtype

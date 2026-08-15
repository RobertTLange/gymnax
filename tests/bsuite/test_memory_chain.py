"""Regression tests for the bsuite MemoryChain environment."""

import jax
import pytest

from gymnax.environments.bsuite.memory_chain import MemoryChain


@pytest.mark.parametrize("num_bits", [1, 2, 4])
def test_memory_chain_resets_and_steps_when_jitted(num_bits):
    """MemoryChain preserves vector context observations for every bit width."""
    env = MemoryChain(num_bits=num_bits)
    params = env.default_params
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    key, reset_key, step_key = jax.random.split(jax.random.key(0), 3)

    observation, state = reset(reset_key, params)
    next_observation, next_state, _, _, _ = step(step_key, state, 0, params)

    assert observation.shape == (num_bits + 2,)
    assert state.context.shape == (num_bits,)
    assert next_observation.shape == (num_bits + 2,)
    assert next_state.context.shape == (num_bits,)
    jax.block_until_ready((next_observation, next_state))

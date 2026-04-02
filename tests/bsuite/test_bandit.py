"""Regression tests for the bsuite bandit environment."""

import jax
import jax.numpy as jnp

from gymnax.environments.bsuite.bandit import SimpleBandit


def test_simple_bandit_reset_initializes_named_fields():
    """Reset should populate rewards, regret, and time in the right slots."""
    env = SimpleBandit(num_actions=5)

    _, state = env.reset_env(jax.random.key(0), env.default_params)

    assert state.rewards.shape == (5,)
    assert jnp.isclose(state.total_regret, 0.0)
    assert state.time == 0
    assert jnp.asarray(state.total_regret).shape == ()
    assert jnp.asarray(state.time).shape == ()


def test_simple_bandit_step_updates_regret_and_time():
    """Step should preserve rewards and increment regret/time."""
    env = SimpleBandit(num_actions=5)
    params = env.default_params

    _, state = env.reset_env(jax.random.key(0), params)
    action = 0

    _, next_state, reward, done, _ = env.step_env(
        jax.random.key(1), state, action, params
    )

    assert jnp.array_equal(next_state.rewards, state.rewards)
    assert jnp.isclose(reward, state.rewards[action])
    assert jnp.isclose(next_state.total_regret, params.optimal_return - reward)
    assert next_state.time == 1
    assert done
    assert jnp.asarray(next_state.total_regret).shape == ()
    assert jnp.asarray(next_state.time).shape == ()

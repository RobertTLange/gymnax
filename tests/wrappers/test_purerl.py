"""Regression tests for PureRL wrappers."""

import jax
import jax.numpy as jnp

import gymnax
from gymnax.wrappers import LogWrapper


def test_log_wrapper_reset_and_step_share_jit_state_dtypes():
    """Catch log states can flow through reset-or-step control flow."""
    env, params = gymnax.make("Catch-bsuite")
    wrapper = LogWrapper(env)
    reset_key, action_key, step_key = jax.random.split(jax.random.key(0), 3)
    _, state = wrapper.reset(reset_key, params)
    action = env.action_space(params).sample(action_key)

    @jax.jit
    def reset_or_step(should_reset, transition_key, log_state):
        return jax.lax.cond(
            should_reset,
            lambda _: wrapper.reset(transition_key, params)[1],
            lambda _: wrapper.step(transition_key, log_state, action, params)[1],
            operand=None,
        )

    for should_reset in (True, False):
        next_state = reset_or_step(jnp.array(should_reset), step_key, state)

        assert next_state.episode_returns.dtype == jnp.float32
        assert next_state.returned_episode_returns.dtype == jnp.float32
        assert next_state.episode_lengths.dtype == jnp.int32
        assert next_state.returned_episode_lengths.dtype == jnp.int32

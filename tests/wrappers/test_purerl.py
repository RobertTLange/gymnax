"""Regression tests for PureRL wrappers."""

import jax
import jax.numpy as jnp
import pytest

import gymnax
from gymnax.wrappers import FlattenObservationWrapper, LogWrapper, StickyActionWrapper


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


def test_sticky_action_wrapper_defaults_to_passthrough_and_records_action():
    env, params = gymnax.make("Catch-bsuite")
    wrapper = StickyActionWrapper(env)
    reset_key, step_key = jax.random.split(jax.random.key(0))
    _, state = wrapper.reset(reset_key, params)

    _, direct_state = env.reset(reset_key, params)
    _, expected_state, expected_reward, expected_terminated, expected_truncated, _ = (
        env.step(
            step_key, direct_state, jnp.int32(2), params
        )
    )
    _, state, reward, terminated, truncated, _ = wrapper.step(
        step_key, state, jnp.int32(2), params
    )

    assert state.last_action == 2
    assert state.env_state.paddle_x == 3
    state_matches = jax.tree.map(jnp.array_equal, state.env_state, expected_state)
    assert all(jax.tree.leaves(state_matches))
    assert reward == expected_reward
    assert terminated == expected_terminated
    assert truncated == expected_truncated


def test_sticky_action_wrapper_replays_previous_action():
    env, params = gymnax.make("Catch-bsuite")
    wrapper = StickyActionWrapper(env, sticky_action_prob=1.0)
    reset_key, step_key = jax.random.split(jax.random.key(1))
    _, state = wrapper.reset(reset_key, params)
    state = state.replace(last_action=jnp.int32(2))

    _, state, _, _, _, _ = wrapper.step(step_key, state, jnp.int32(0), params)

    assert state.last_action == 2
    assert state.env_state.paddle_x == 3


def test_sticky_action_wrapper_samples_replay_from_transition_key():
    env, params = gymnax.make("Catch-bsuite")
    wrapper = StickyActionWrapper(env, sticky_action_prob=0.5)
    reset_key, step_key = jax.random.split(jax.random.key(2))
    _, state = wrapper.reset(reset_key, params)

    _, state, _, _, _, _ = wrapper.step(step_key, state, jnp.int32(2), params)
    replay_key, _ = jax.random.split(step_key)
    expected_action = jnp.where(
        jax.random.bernoulli(replay_key, 0.5), jnp.int32(0), jnp.int32(2)
    )

    assert state.last_action == expected_action
    assert state.env_state.paddle_x == jnp.where(expected_action == 0, 1, 3)


def test_sticky_action_wrapper_keeps_last_action_after_auto_reset():
    env, _ = gymnax.make("Catch-bsuite")
    params = env.default_params.replace(max_steps_in_episode=1)
    wrapper = StickyActionWrapper(env, sticky_action_prob=1.0)
    reset_key, step_key = jax.random.split(jax.random.key(3))
    _, state = wrapper.reset(reset_key, params)
    state = state.replace(last_action=jnp.int32(2))

    _, state, _, terminated, truncated, _ = wrapper.step(
        step_key, state, jnp.int32(0), params
    )
    done = jnp.logical_or(terminated, truncated)

    assert done
    assert state.last_action == 2
    assert state.env_state.time == 0


def test_sticky_action_wrapper_explicit_reset_reinitializes_last_action():
    env, params = gymnax.make("Catch-bsuite")
    wrapper = StickyActionWrapper(env)
    reset_key, step_key, fresh_reset_key = jax.random.split(jax.random.key(4), 3)
    _, state = wrapper.reset(reset_key, params)
    _, state, _, _, _, _ = wrapper.step(step_key, state, jnp.int32(2), params)

    _, fresh_state = wrapper.reset(fresh_reset_key, params)

    assert fresh_state.last_action == 0


def test_flatten_observation_wrapper_flattens_terminal_observation():
    """Bootstrap observations remain in the wrapper's flattened observation space."""
    env, params = gymnax.make("Breakout-MinAtar")
    params = params.replace(max_steps_in_episode=1)
    wrapper = FlattenObservationWrapper(env)
    reset_key, action_key, step_key = jax.random.split(jax.random.key(5), 3)
    _, state = wrapper.reset(reset_key, params)
    action = env.action_space(params).sample(action_key)

    obs, _, _, _, truncated, info = wrapper.step(step_key, state, action, params)

    assert truncated
    assert info["final_observation"].shape == obs.shape


def test_sticky_action_wrapper_supports_jit_and_vmap():
    env, params = gymnax.make("Catch-bsuite")
    wrapper = StickyActionWrapper(env)
    reset_keys = jax.random.split(jax.random.key(6), 2)
    step_keys = jax.random.split(jax.random.key(7), 2)
    actions = jnp.array([0, 2], dtype=jnp.int32)

    _, states = jax.jit(jax.vmap(wrapper.reset, in_axes=(0, None)))(reset_keys, params)
    _, states, _, _, _, _ = jax.jit(jax.vmap(wrapper.step, in_axes=(0, 0, 0, None)))(
        step_keys, states, actions, params
    )

    assert jnp.array_equal(states.last_action, actions)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_sticky_action_wrapper_rejects_invalid_probability(probability):
    env, _ = gymnax.make("Catch-bsuite")

    with pytest.raises(ValueError, match="sticky_action_prob"):
        StickyActionWrapper(env, sticky_action_prob=probability)


def test_sticky_action_wrapper_requires_discrete_actions():
    env, _ = gymnax.make("Pendulum-v1")

    with pytest.raises(ValueError, match="Discrete"):
        StickyActionWrapper(env)

"""Regression tests for the 0.x terminal-transition metadata contract."""

import jax
import jax.numpy as jnp

import gymnax
from gymnax.environments import environment


class LegacyActionTerminalEnvironment(
    environment.Environment[environment.EnvState, environment.EnvParams]
):
    """Legacy environment whose terminal result depends on the action."""

    def step_env(self, key, state, action, params):
        del key, params
        next_state = state.replace(time=state.time + 1)
        return (
            jnp.array([next_state.time]),
            next_state,
            jnp.array(0.0),
            jnp.asarray(action, dtype=jnp.bool),
            {},
        )

    def reset_env(self, key, params):
        del key, params
        return jnp.array([-1]), environment.EnvState(time=jnp.int32(0))

    def is_terminal(self, state, params):
        return state.time >= params.max_steps_in_episode


def _step_at_time_limit(env_name: str):
    env, params = gymnax.make(env_name)
    params = params.replace(max_steps_in_episode=1)
    key, reset_key, action_key, step_key = jax.random.split(jax.random.key(0), 4)
    _, state = env.reset(reset_key, params)
    action = env.action_space(params).sample(action_key)
    return env, params, state, action, step_key


def test_time_limit_exposes_terminal_metadata_and_legacy_discount():
    """A time limit resets while retaining the pre-reset bootstrap observation."""
    env, params, state, action, step_key = _step_at_time_limit("Pendulum-v1")
    transition_key, _ = jax.random.split(step_key)
    final_observation, _, _, _, _ = env.step_env(transition_key, state, action, params)

    observation, next_state, _, done, info = env.step(step_key, state, action, params)

    assert done
    assert not info["terminated"]
    assert info["truncated"]
    assert next_state.time == 0
    assert not jnp.array_equal(observation, final_observation)
    assert jnp.array_equal(info["final_observation"], final_observation)
    assert info["discount"] == 0.0


def test_natural_termination_is_distinct_from_time_limit():
    """A CartPole failure before its time limit is a natural termination."""
    env, params = gymnax.make("CartPole-v1")
    params = params.replace(max_steps_in_episode=2)
    key, reset_key, step_key = jax.random.split(jax.random.key(1), 3)
    _, state = env.reset(reset_key, params)
    state = state.replace(theta=params.theta_threshold_radians * 2)

    _, _, _, done, info = env.step(step_key, state, jnp.int32(0), params)

    assert done
    assert info["terminated"]
    assert not info["truncated"]


def test_terminal_transition_can_be_both_terminated_and_truncated():
    """Natural termination at the horizon reports both causes."""
    env, params = gymnax.make("CartPole-v1")
    params = params.replace(max_steps_in_episode=1)
    key, reset_key, step_key = jax.random.split(jax.random.key(2), 3)
    _, state = env.reset(reset_key, params)
    state = state.replace(theta=params.theta_threshold_radians * 2)

    _, _, _, done, info = env.step(step_key, state, jnp.int32(0), params)

    assert done
    assert info["terminated"]
    assert info["truncated"]


def test_terminal_metadata_has_stable_jit_and_vmap_structure():
    """Terminal metadata works through batched compiled time-limit transitions."""
    env, params, _, _, _ = _step_at_time_limit("Pendulum-v1")

    def reset_and_step(key):
        reset_key, action_key, step_key = jax.random.split(key, 3)
        _, state = env.reset(reset_key, params)
        action = env.action_space(params).sample(action_key)
        return env.step(step_key, state, action, params)

    _, next_states, _, done, info = jax.jit(jax.vmap(reset_and_step))(
        jax.random.split(jax.random.key(3), 2)
    )

    assert jnp.all(done)
    assert jnp.all(info["truncated"])
    assert not jnp.any(info["terminated"])
    assert info["final_observation"].shape[0] == 2
    assert jnp.all(next_states.time == 0)


def test_legacy_transition_terminal_result_still_controls_autoreset():
    """Legacy custom environments retain their pre-existing reset behavior."""
    env = LegacyActionTerminalEnvironment()
    params = environment.EnvParams(max_steps_in_episode=2)
    _, state = env.reset(jax.random.key(4), params)

    _, next_state, _, done, info = env.step(
        jax.random.key(5), state, jnp.array(True), params
    )

    assert done
    assert next_state.time == 0
    assert info["terminated"]
    assert not info["truncated"]

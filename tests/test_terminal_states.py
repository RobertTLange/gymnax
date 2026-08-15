"""Regression tests for the 1.0 terminal-transition contract."""

import jax
import jax.numpy as jnp

import gymnax
from gymnax.environments import environment
from gymnax.wrappers import LegacyStepAPIWrapper


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

    def observe(self, key, state, action, params):
        del key, action, params
        return jnp.array([state.time])


class PyTreeObservationEnvironment(
    environment.Environment[environment.EnvState, environment.EnvParams]
):
    """Small environment exercising structured observations and autoreset."""

    def step_env(self, key, state, action, params):
        next_state = state.replace(time=state.time + 1)
        return (
            self.observe(key, next_state, action, params),
            next_state,
            jnp.array(0.0),
            jnp.array(False),
            {},
        )

    def reset_env(self, key, params):
        state = environment.EnvState(time=jnp.int32(0))
        return self.observe(key, state, None, params), state

    def observe(self, key, state, action, params):
        del key, action, params
        return {"time": jnp.array([state.time]), "constant": (jnp.array(1),)}


def _step_at_time_limit(env_name: str):
    env, params = gymnax.make(env_name)
    params = params.replace(max_steps_in_episode=1)
    key, reset_key, action_key, step_key = jax.random.split(jax.random.key(0), 4)
    _, state = env.reset(reset_key, params)
    action = env.action_space(params).sample(action_key)
    return env, params, state, action, step_key


def test_time_limit_returns_distinct_terminal_flags():
    """A time limit resets while retaining the pre-reset bootstrap observation."""
    env, params, state, action, step_key = _step_at_time_limit("Pendulum-v1")
    transition_key, _ = jax.random.split(step_key)
    final_observation, _, _, _, _ = env.step_env(transition_key, state, action, params)

    observation, next_state, _, terminated, truncated, info = env.step(
        step_key, state, action, params
    )

    assert not terminated
    assert truncated
    assert next_state.time == 0
    assert not jnp.array_equal(observation, final_observation)
    assert jnp.array_equal(info["final_observation"], final_observation)


def test_natural_termination_is_distinct_from_time_limit():
    """A CartPole failure before its time limit is a natural termination."""
    env, params = gymnax.make("CartPole-v1")
    params = params.replace(max_steps_in_episode=2)
    key, reset_key, step_key = jax.random.split(jax.random.key(1), 3)
    _, state = env.reset(reset_key, params)
    state = state.replace(theta=params.theta_threshold_radians * 2)

    _, _, _, terminated, truncated, _ = env.step(step_key, state, jnp.int32(0), params)

    assert terminated
    assert not truncated


def test_terminal_transition_can_be_both_terminated_and_truncated():
    """Natural termination at the horizon reports both causes."""
    env, params = gymnax.make("CartPole-v1")
    params = params.replace(max_steps_in_episode=1)
    key, reset_key, step_key = jax.random.split(jax.random.key(2), 3)
    _, state = env.reset(reset_key, params)
    state = state.replace(theta=params.theta_threshold_radians * 2)

    _, _, _, terminated, truncated, _ = env.step(step_key, state, jnp.int32(0), params)

    assert terminated
    assert truncated


def test_terminal_metadata_has_stable_jit_and_vmap_structure():
    """Terminal metadata works through batched compiled time-limit transitions."""
    env, params, _, _, _ = _step_at_time_limit("Pendulum-v1")

    def reset_and_step(key):
        reset_key, action_key, step_key = jax.random.split(key, 3)
        _, state = env.reset(reset_key, params)
        action = env.action_space(params).sample(action_key)
        return env.step(step_key, state, action, params)

    _, next_states, _, terminated, truncated, info = jax.jit(jax.vmap(reset_and_step))(
        jax.random.split(jax.random.key(3), 2)
    )

    assert jnp.all(truncated)
    assert not jnp.any(terminated)
    assert info["final_observation"].shape[0] == 2
    assert jnp.all(next_states.time == 0)


def test_step_env_terminated_result_controls_autoreset():
    """Action-dependent termination controls autoreset directly."""
    env = LegacyActionTerminalEnvironment()
    params = environment.EnvParams(max_steps_in_episode=2)
    _, state = env.reset(jax.random.key(4), params)

    _, next_state, _, terminated, truncated, _ = env.step(
        jax.random.key(5), state, jnp.array(True), params
    )

    assert terminated
    assert not truncated
    assert next_state.time == 0


def test_legacy_wrapper_restores_five_value_step_api():
    """The opt-in adapter exposes a 0.x-compatible done flag."""
    env, params, state, action, step_key = _step_at_time_limit("Pendulum-v1")

    _, next_state, _, done, info = LegacyStepAPIWrapper(env).step(
        step_key, state, action, params
    )

    assert done
    assert next_state.time == 0
    assert info["truncated"]


def test_pytree_observations_survive_autoreset_jit_and_vmap():
    """Structured observations retain their tree shape at the time limit."""
    env = PyTreeObservationEnvironment()
    params = environment.EnvParams(max_steps_in_episode=1)

    def reset_and_step(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)
        return env.step(step_key, state, jnp.int32(0), params)

    obs, states, _, terminated, truncated, info = jax.jit(jax.vmap(reset_and_step))(
        jax.random.split(jax.random.key(6), 2)
    )

    assert jnp.all(truncated)
    assert not jnp.any(terminated)
    assert obs["time"].shape == (2, 1)
    assert info["final_observation"]["time"].shape == (2, 1)
    assert jnp.all(states.time == 0)

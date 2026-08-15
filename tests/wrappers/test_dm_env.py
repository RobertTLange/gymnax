"""Tests for the GymnaxToDmEnvWrapper."""

import chex
import jax
import jax.numpy as jnp

import gymnax
from gymnax.wrappers import dm_env


def test_dmenv_wrapper():
    """Wrap a Gymnax environment in dm_env style."""
    env, env_params = gymnax.make("CartPole-v1")
    wrapped_env = dm_env.GymnaxToDmEnvWrapper(env)
    keys = jax.random.split(jax.random.key(0), 16)
    action = jax.vmap(env.action_space(env_params).sample)(keys)
    _, env_state = jax.vmap(env.reset)(keys)
    o, _, r, terminated, truncated, _ = jax.vmap(env.step)(keys, env_state, action)
    d = jax.numpy.logical_or(terminated, truncated)

    reset_fn = jax.vmap(wrapped_env.reset, in_axes=(0,))
    timesteps = reset_fn(keys)
    chex.assert_trees_all_equal_shapes(o, timesteps.observation)
    chex.assert_trees_all_equal_shapes(r, timesteps.reward)
    chex.assert_trees_all_equal_shapes(d, timesteps.discount)

    step_fn = jax.vmap(wrapped_env.step, in_axes=(0, 0, 0))
    _ = step_fn(keys, timesteps, action)


def test_dmenv_wrapper_keeps_bootstrap_discount_at_time_limit():
    """DM-env discount remains one when an episode ends by truncation."""
    env, params = gymnax.make("Pendulum-v1")
    params = params.replace(max_steps_in_episode=1)
    wrapped_env = dm_env.GymnaxToDmEnvWrapper(env)
    reset_key, action_key, step_key = jax.random.split(jax.random.key(1), 3)

    _, state = env.reset(reset_key, params)
    action = env.action_space(params).sample(action_key)
    _, _, _, _, truncated, info = env.step(step_key, state, action, params)
    timestep = wrapped_env.reset(reset_key, params)
    next_timestep = wrapped_env.step(step_key, timestep, action, params)

    assert truncated
    assert next_timestep.discount == jnp.array(1.0)
    assert jnp.array_equal(next_timestep.final_observation, info["final_observation"])

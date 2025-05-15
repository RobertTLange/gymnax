"""Tests for the GymnaxToDmEnvWrapper."""

import chex
import jax

import gymnax
from gymnax.wrappers import dm_env


def test_dmenv_wrapper():
    """Wrap a Gymnax environment in dm_env style."""
    env, env_params = gymnax.make("CartPole-v1")
    wrapped_env = dm_env.GymnaxToDmEnvWrapper(env)
    keys = jax.random.split(jax.random.key(0), 16)
    action = jax.vmap(env.action_space(env_params).sample)(keys)
    _, env_state = jax.vmap(env.reset)(keys)
    o, _, r, d, _ = jax.vmap(env.step)(keys, env_state, action)

    reset_fn = jax.vmap(wrapped_env.reset, in_axes=(0,))
    timesteps = reset_fn(keys)
    chex.assert_trees_all_equal_shapes(o, timesteps.observation)
    chex.assert_trees_all_equal_shapes(r, timesteps.reward)
    chex.assert_trees_all_equal_shapes(d, timesteps.discount)

    step_fn = jax.vmap(wrapped_env.step, in_axes=(0, 0, 0))
    _ = step_fn(keys, timesteps, action)

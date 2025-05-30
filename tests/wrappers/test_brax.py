"""Tests for brax wrapper."""

import chex
import jax
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from brax import envs

import gymnax
from gymnax.wrappers import brax


def test_brax_wrapper():
    """Wrap a Gymnax environment in brax.


    Use Brax's wrappers to handle vmap and episodes.
    """
    env, env_params = gymnax.make("CartPole-v1")
    brax_env = brax.GymnaxToBraxWrapper(env)
    wrapped_env = envs.wrappers.training.VmapWrapper(
        envs.wrappers.training.EpisodeWrapper(brax_env, 100, 1)
    )
    b = 16
    keys = jax.random.split(jax.random.key(0), b)
    action = jax.vmap(env.action_space(env_params).sample)(keys)
    reset_fn = jax.jit(wrapped_env.reset)
    _, env_state = jax.vmap(env.reset)(keys)
    o, new_env_state, r, d, _ = jax.vmap(env.step)(keys, env_state, action)
    state = reset_fn(keys)
    chex.assert_trees_all_equal_shapes(o, state.obs)
    chex.assert_trees_all_equal_shapes(r, state.reward)
    chex.assert_trees_all_equal_shapes(d, state.done)
    chex.assert_trees_all_equal_structs(new_env_state, state.pipeline_state)
    step_fn = jax.jit(wrapped_env.step)
    _ = step_fn(state, action)

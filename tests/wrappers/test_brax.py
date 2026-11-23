"""Tests for brax wrapper."""

import warnings

import chex
import jax
import pytest

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
    o, new_env_state, r, terminated, truncated, _ = jax.vmap(env.step)(
        keys, env_state, action
    )
    d = jax.numpy.logical_or(terminated, truncated)
    state = reset_fn(keys)
    chex.assert_trees_all_equal_shapes(o, state.obs)
    chex.assert_trees_all_equal_shapes(r, state.reward)
    chex.assert_trees_all_equal_shapes(d, state.done)
    chex.assert_trees_all_equal_structs(new_env_state, state.pipeline_state)
    step_fn = jax.jit(wrapped_env.step)
    _ = step_fn(state, action)


def test_brax_wrapper_retains_terminal_flags_in_info():
    """Brax exposes Gymnax terminal causes alongside its combined done flag."""
    env, params = gymnax.make("Pendulum-v1")
    params = params.replace(max_steps_in_episode=1)
    adapter = brax.GymnaxToBraxWrapper(env)
    reset_key, action_key = jax.random.split(jax.random.key(1))

    state = adapter.reset(reset_key, params)
    action = env.action_space(params).sample(action_key)
    next_state = adapter.step(state, action)

    assert next_state.done
    assert not next_state.info["terminated"]
    assert next_state.info["truncated"]


@pytest.mark.parametrize(
    ("env_name", "action_size", "observation_size"),
    [
        ("CartPole-v1", 1, 4),
        ("Pendulum-v1", 1, 3),
        ("PointRobot-misc", 2, 6),
        ("Reacher-misc", 2, 8),
        ("Swimmer-misc", 2, 16),
        ("Snake-misc", 1, 3),
    ],
)
def test_brax_wrapper_size_properties(env_name, action_size, observation_size):
    env, _ = gymnax.make(env_name)
    adapter = brax.GymnaxToBraxWrapper(env)

    assert adapter.action_size == action_size
    assert adapter.observation_size == observation_size
    assert adapter.backend == "jax"


def test_brax_wrapper_ppo_network_initialization():
    from brax.training.agents.ppo import networks

    env, _ = gymnax.make("PointRobot-misc")
    adapter = brax.GymnaxToBraxWrapper(env)
    wrapped = envs.wrappers.training.EpisodeWrapper(adapter, 100, 1)
    network = networks.make_ppo_networks(wrapped.observation_size, wrapped.action_size)
    policy_params = network.policy_network.init(jax.random.key(0))
    value_params = network.value_network.init(jax.random.key(1))
    state = adapter.reset(jax.random.key(2))

    logits = network.policy_network.apply(None, policy_params, state.obs)
    value = network.value_network.apply(None, value_params, state.obs)
    assert logits.shape == (4,)
    assert value.shape == ()
    action = network.parametric_action_distribution.sample(logits, jax.random.key(3))
    assert action.shape == (2,)
    next_state = jax.jit(adapter.step)(state, action)
    assert next_state.obs.shape == state.obs.shape

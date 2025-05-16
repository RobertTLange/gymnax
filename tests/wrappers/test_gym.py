"""Tests for the GymnaxToGymWrapper and GymnaxToVectorGymWrapper."""

import chex
import gymnasium as gym
import jax

import gymnax
from gymnax.wrappers import gym as wrappers


def test_gym_wrapper():
    """Test the GymnaxToGymWrapper."""
    env, env_params = gymnax.make("FourRooms-misc")
    e = wrappers.GymnaxToGymWrapper(env, env_params, 0)
    a_space, _ = e.action_space, e.observation_space
    o, _ = e.reset()
    next_keys = jax.random.split(jax.random.key(0), 2)
    o_env, env_state = env.reset(next_keys[1], env_params)
    chex.assert_trees_all_close(o, o_env)
    chex.assert_trees_all_close(env_state.__dict__, e.env_state.__dict__)
    chex.assert_trees_all_close(
        jax.random.key_data(next_keys[0]), jax.random.key_data(e.key)
    )
    e.reset(seed=5)
    next_keys = jax.random.split(jax.random.key(5), 2)
    chex.assert_trees_all_close(
        jax.random.key_data(next_keys[0]), jax.random.key_data(e.key)
    )
    _, _, _, _, _ = e.step(a_space.sample())
    e.render()


def test_gym_vector_wrapper():
    """Test the GymnaxToVectorGymWrapper."""
    env, env_params = gymnax.make("FourRooms-misc")
    b = 16  # 16 parallel envs
    e = wrappers.GymnaxToVectorGymWrapper(env, b, env_params, 0)
    a_space, o_space = e.action_space, e.observation_space
    assert isinstance(a_space, gym.spaces.MultiDiscrete)
    assert isinstance(o_space, gym.spaces.Box)
    single_a_space, single_o_space = (
        e.single_action_space,
        e.single_observation_space,
    )
    assert isinstance(single_a_space, gym.spaces.Discrete)
    assert isinstance(single_o_space, gym.spaces.Box)
    keys = jax.random.split(jax.random.key(0), b)
    chex.assert_trees_all_close(jax.random.key_data(e.key), jax.random.key_data(keys))
    o, _ = e.reset()
    env_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    _ = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    o_env, _ = env_reset(keys, env_params)
    chex.assert_trees_all_equal_shapes(o_env, o)
    _, _, _, _, _ = e.step(e.action_space.sample())
    e.render()

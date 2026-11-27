"""Tests for the GymnaxToGymWrapper and GymnaxToVectorGymWrapper."""

import copy
from types import MappingProxyType

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

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


def test_gym_wrapper_forwards_time_limit_metadata():
    """The Gymnasium adapter separates a Gymnax time limit from termination."""
    env, params = gymnax.make("Pendulum-v1")
    params = params.replace(max_steps_in_episode=1)
    wrapped = wrappers.GymnaxToGymWrapper(env, params, 0)
    wrapped.reset()

    observation, _, terminated, truncated, info = wrapped.step(jnp.zeros((1,)))

    assert not terminated
    assert truncated
    assert "final_observation" in info
    assert info["final_observation"].shape == observation.shape


def test_gym_wrapper_forwards_natural_termination_metadata():
    """The Gymnasium adapter preserves a CartPole natural termination."""
    env, params = gymnax.make("CartPole-v1")
    params = params.replace(max_steps_in_episode=2)
    wrapped = wrappers.GymnaxToGymWrapper(env, params, 0)
    wrapped.reset()
    wrapped.env_state = wrapped.env_state.replace(
        theta=params.theta_threshold_radians * 2
    )

    _, _, terminated, truncated, _ = wrapped.step(jnp.int32(0))

    assert terminated
    assert not truncated


def test_vector_gym_wrapper_forwards_terminal_metadata():
    """The vector adapter forwards a batch of time-limit flags and observations."""
    env, params = gymnax.make("Pendulum-v1")
    params = params.replace(max_steps_in_episode=1)
    wrapped = wrappers.GymnaxToVectorGymWrapper(env, 2, params, 0)
    wrapped.reset()

    observation, _, terminated, truncated, info = wrapped.step(
        jnp.zeros((2, 1), dtype=jnp.float32)
    )

    assert not jnp.any(terminated)
    assert jnp.all(truncated)
    assert info["final_observation"].shape == observation.shape


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("CartPole-v1", "Pendulum-v1"),
        ("FrozenLake-misc", "CartPole-v1"),
        ("Pong-misc", "FrozenLake-misc"),
    ],
)
def test_gym_wrapper_metadata_is_independent(first_name, second_name):
    original_metadata = copy.deepcopy(gym.Env.metadata)
    try:
        first = wrappers.GymnaxToGymWrapper(gymnax.make(first_name)[0])
        second = wrappers.GymnaxToGymWrapper(gymnax.make(second_name)[0])

        assert first.metadata["name"] == first_name
        assert second.metadata["name"] == second_name
        assert first.metadata is not second.metadata
        assert gym.Env.metadata == original_metadata

        first.metadata["render_modes"].append("custom")
        assert "custom" not in second.metadata["render_modes"]
        assert gym.Env.metadata == original_metadata
    finally:
        gym.Env.metadata.clear()
        gym.Env.metadata.update(original_metadata)


@pytest.mark.parametrize("renderable", [False, True])
def test_gym_wrapper_preserves_inherited_metadata(renderable):
    settings = MappingProxyType({"values": (1,)})

    class CustomWrapper(wrappers.GymnaxToGymWrapper):
        metadata = {"render_fps": 25, "custom": settings}

    env = gymnax.make("CartPole-v1")[0]
    if renderable:
        env.render = lambda state, params: None
    original_metadata = CustomWrapper.metadata.copy()
    wrapped = CustomWrapper(env)

    assert wrapped.metadata["render_modes"] == (
        ["human", "rgb_array"] if renderable else []
    )
    assert wrapped.metadata["render_fps"] == 25
    assert wrapped.metadata["custom"] is settings
    assert CustomWrapper.metadata == original_metadata

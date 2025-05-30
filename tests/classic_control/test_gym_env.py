"""Tests for the classic control environments."""

import gymnasium as gym
import jax

import gymnax
from tests import helpers, state_translate

num_episodes, num_steps, tolerance = 10, 150, 1e-4


def test_step(gym_env_name):
    """Test a step transition for the env."""
    key = jax.random.key(0)

    # Instantiate gymnasium and gymnax environments
    env_gym = gym.make(gym_env_name).unwrapped
    env_gymnax, env_params = gymnax.make(gym_env_name)

    # Loop over test episodes
    for _ in range(num_episodes):
        _ = env_gym.reset()
        # Loop over test episode steps
        for _ in range(num_steps):
            action = env_gym.action_space.sample()
            state = state_translate.np_state_to_jax(env_gym, gym_env_name, get_jax=True)
            obs_gym, reward_gym, done_gym, _, _ = env_gym.step(action)

            key, key_input = jax.random.split(key)
            obs_jax, state_jax, reward_jax, done_jax, _ = env_gymnax.step(
                key_input, state, action, env_params
            )

            # Check correctness of transition
            helpers.assert_correct_transit(
                obs_gym,
                reward_gym,
                done_gym,
                obs_jax,
                reward_jax,
                done_jax,
                tolerance,
            )

            # Check that post-transition states are equal
            if not done_gym:
                helpers.assert_correct_state(
                    env_gym, gym_env_name, state_jax, tolerance
                )
            else:
                break


def test_reset(gym_env_name):
    """Test reset obs/state is in space of OpenAI version."""
    key = jax.random.key(0)

    # Instantiate gymnax environment
    env_gymnax, env_params = gymnax.make(gym_env_name)

    for _ in range(num_episodes):
        key, key_input = jax.random.split(key)
        obs, state = env_gymnax.reset(key_input, env_params)

        # Check state and observation space
        env_gymnax.state_space(env_params).contains(state)
        env_gymnax.observation_space(env_params).contains(obs)

"""Tests for misc environments."""

import jax

import gymnax

num_episodes, num_steps, tolerance = 10, 100, 1e-04


def test_step(misc_env_name: str):
    """Test a step transition for the env."""
    key = jax.random.key(0)

    # Instantiate gymnax environment
    env_gymnax, env_params = gymnax.make(misc_env_name)

    # Loop over test episodes
    for _ in range(num_episodes):
        key, key_input = jax.random.split(key)
        obs, state = env_gymnax.reset(key_input, env_params)
        # Loop over test episode steps
        for _ in range(num_steps):
            action = env_gymnax.action_space(env_params).sample(key_input)
            obs, state, _, _, _ = env_gymnax.step(key_input, state, action, env_params)

            # Check state and observation space
            env_gymnax.state_space(env_params).contains(state)
            env_gymnax.observation_space(env_params).contains(obs)


def test_reset(misc_env_name: str):
    """Test reset obs/state is in space of OpenAI version."""
    key = jax.random.key(0)

    # Instantiate gymnax environment
    env_gymnax, env_params = gymnax.make(misc_env_name)

    for _ in range(num_episodes):
        key, key_input = jax.random.split(key)
        obs, state = env_gymnax.reset(key_input, env_params)

        # Check state and observation space
        env_gymnax.state_space(env_params).contains(state)
        env_gymnax.observation_space(env_params).contains(obs)

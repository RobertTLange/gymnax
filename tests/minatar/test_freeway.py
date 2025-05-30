"""Tests for the Freeway environment."""

import warnings
import traceback

_original_warn = warnings.warn


def custom_warn(message, category=None, stacklevel=1, source=None):
    print("\n=== WARNING ===")
    print(f"{category.__name__ if category else 'Warning'}: {message}")
    traceback.print_stack()
    _original_warn(message, category, stacklevel, source)


import freeway_helpers
import jax
import jax.numpy as jnp
import numpy as np
from minatar import environment

import gymnax
from gymnax.environments.minatar import freeway
from tests import helpers, state_translate

num_episodes, num_steps, tolerance = 5, 10, 1e-04
env_name_gym, env_name_jax = "freeway", "Freeway-MinAtar"


def test_step():
    """Test a step transition for the env."""
    key = jax.random.key(0)
    env_gym = environment.Environment(env_name_gym, sticky_action_prob=0.0)
    env_gymnax, env_params = gymnax.make(env_name_jax)

    # Loop over test episodes
    for _ in range(num_episodes):
        _ = env_gym.reset()
        # Loop over test episode steps
        for _ in range(num_steps):
            key, _, key_action = jax.random.split(key, 3)
            state = state_translate.np_state_to_jax(env_gym, env_name_jax, get_jax=True)
            action = env_gymnax.action_space(env_params).sample(key_action)
            action_gym = helpers.minatar_action_map(action, env_name_jax)

            # Perform step transition for agent & assert correct state dict
            _ = freeway_helpers.step_agent_numpy(env_gym, action_gym)
            state_jax_post_agent, _, _ = freeway.step_agent(
                action_gym, state, env_params
            )
            helpers.assert_correct_state(
                env_gym, env_name_jax, state_jax_post_agent, tolerance
            )

            # Perform step transition for cars & assert correct state dict
            state_gym_post_agent = state_translate.np_state_to_jax(
                env_gym, env_name_jax, get_jax=True
            )
            freeway_helpers.step_cars_numpy(env_gym)
            state_jax_post_cars = freeway.step_cars(state_gym_post_agent)
            helpers.assert_correct_state(
                env_gym, env_name_jax, state_jax_post_cars, tolerance
            )

            if env_gym.env.terminal:
                break


def test_reset():
    """Test reset obs/state is in space of NumPy version."""
    # env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    key = jax.random.key(0)
    env_gymnax, env_params = gymnax.make(env_name_jax)
    for _ in range(num_episodes):
        key, key_input = jax.random.split(key)
        obs, state = env_gymnax.reset(key_input, env_params)
        # Check state and observation space
        env_gymnax.state_space(env_params).contains(state)
        env_gymnax.observation_space(env_params).contains(obs)


def test_get_obs():
    """Test observation function."""
    key = jax.random.key(0)
    env_gym = environment.Environment(env_name_gym, sticky_action_prob=0.0)
    env_gymnax, env_params = gymnax.make(env_name_jax)

    # Loop over test episodes
    for _ in range(num_episodes):
        env_gym.reset()
        # Loop over test episode steps
        for _ in range(num_steps):
            key, _, key_action = jax.random.split(key, 3)
            action = env_gymnax.action_space(env_params).sample(key_action)
            action_gym = helpers.minatar_action_map(action, env_name_jax)
            # Step gym environment get state and trafo in jax dict
            _ = env_gym.act(action_gym)
            obs_gym = env_gym.state()
            state = state_translate.np_state_to_jax(env_gym, env_name_jax, get_jax=True)
            obs_jax = env_gymnax.get_obs(state)
            # Check for correctness of observations
            assert (obs_gym == obs_jax).all()
            done_gym = env_gym.env.terminal
            # Start a new episode if the previous one has terminated
            if done_gym:
                break


def test_randomize_cars():
    """Test randomization of car locations."""
    # Test initialization version of `randomize_cars`
    for _ in range(num_steps):
        speeds = np.random.randint(1, 6, 8)
        directions = np.random.choice([-1, 1], 8)
        cars_gym = freeway_helpers.det_randomize_cars_numpy(
            speeds, directions, np.zeros((8, 4)), 1
        )
        cars_jax = freeway.randomize_cars(
            speeds, directions, jnp.zeros((8, 4), dtype=int), True
        )
        assert (np.array(cars_gym) == cars_jax).all()

    # Test no initialization version of `randomize_cars`
    for _ in range(num_steps):
        speeds = np.random.randint(1, 6, 8)
        directions = np.random.choice([-1, 1], 8)
        cars_gym = freeway_helpers.det_randomize_cars_numpy(
            speeds, directions, np.zeros((8, 4)), False
        )
        cars_jax = freeway.randomize_cars(
            speeds, directions, jnp.zeros((8, 4), dtype=int), False
        )
        assert (np.array(cars_gym) == cars_jax).all()
    return

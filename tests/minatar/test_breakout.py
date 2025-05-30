"""Tests for the breakout environment."""

import breakout_helpers
import jax
from minatar import environment

import gymnax
from gymnax.environments.minatar import breakout
from tests import helpers, state_translate

num_episodes, num_steps, tolerance = 5, 10, 1e-04
env_name_gym, env_name_jax = "breakout", "Breakout-MinAtar"


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
            key, key_step, key_action = jax.random.split(key, 3)
            state = state_translate.np_state_to_jax(env_gym, env_name_jax, get_jax=True)
            action = env_gymnax.action_space(env_params).sample(key_action)
            action_gym = helpers.minatar_action_map(action, env_name_jax)

            reward_gym, _ = env_gym.act(action_gym)
            obs_gym = env_gym.state()
            done_gym = env_gym.env.terminal
            obs_jax, state_jax, reward_jax, done_jax, _ = env_gymnax.step(
                key_step, state, action, env_params
            )
            # Doesnt make sense to compare since jax resamples state
            if done_gym and done_gym == done_jax:
                break

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
            helpers.assert_correct_state(env_gym, env_name_jax, state_jax, tolerance)


def test_sub_steps():
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

            new_x, new_y = breakout_helpers.step_agent_numpy(env_gym, action_gym)
            state_jax_a, new_x_jax, new_y_jax = breakout.step_agent(state, action_gym)
            assert new_x == new_x_jax and new_y == new_y_jax
            helpers.assert_correct_state(env_gym, env_name_jax, state_jax_a, tolerance)

            _, _ = breakout_helpers.step_ball_brick_numpy(env_gym, new_x, new_y)
            state_jax_b, _ = breakout.step_ball_brick(state_jax_a, new_x, new_y)
            helpers.assert_correct_state(env_gym, env_name_jax, state_jax_b, tolerance)
            if env_gym.env.terminal:
                break


def test_reset():
    """Test reset obs/state is in space of NumPy version."""
    # env_gym = environment.Environment(env_name_gym, sticky_action_prob=0.0)
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

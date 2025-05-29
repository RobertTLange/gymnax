"""Tests for the Space Invaders environment."""

import jax
import space_invaders_helpers
from minatar import environment

import gymnax
from gymnax.environments.minatar import space_invaders
from tests import helpers, state_translate

num_episodes, num_steps, tolerance = 5, 10, 1e-04
env_name_gym, env_name_jax = "space_invaders", "SpaceInvaders-MinAtar"


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

            if done_gym:
                break


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

            _ = space_invaders_helpers.step_agent_numpy(env_gym, action_gym)
            state_jax_a = space_invaders.step_agent(action_gym, state, env_params)
            helpers.assert_correct_state(env_gym, env_name_jax, state_jax_a, tolerance)

            _ = space_invaders_helpers.step_aliens_numpy(env_gym)
            state_jax_b = space_invaders.step_aliens(state_jax_a)
            helpers.assert_correct_state(env_gym, env_name_jax, state_jax_b, tolerance)

            reward_gym = space_invaders_helpers.step_shoot_numpy(env_gym)
            state_jax_c, reward_jax = space_invaders.step_shoot(state_jax_b, env_params)
            helpers.assert_correct_state(env_gym, env_name_jax, state_jax_c, tolerance)
            assert reward_gym == reward_jax
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


def test_nearest_alien():
    """Test nearest alien computation."""
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
            state = state_translate.np_state_to_jax(env_gym, env_name_jax, get_jax=True)

            # Get nearest alien for current environment state
            nearest = space_invaders_helpers.get_nearest_alien_numpy(env_gym)
            alien_exists, loc, idd = space_invaders.get_nearest_alien(
                state.pos, state.alien_map
            )

            if nearest is None:
                assert 1 - alien_exists
            else:
                assert loc == nearest[0]
                assert idd == nearest[1]
            # Start a new episode if the previous one has terminated
            if env_gym.env.terminal:
                break

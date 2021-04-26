import jax
import jax.numpy as jnp
import gymnax
from gymnax.utils import (np_state_to_jax,
                          minatar_action_map,
                          assert_correct_transit,
                          assert_correct_state)

import numpy as np
from minatar import Environment

num_episodes, num_steps, tolerance = 2, 10, 1e-04
env_name_gym, env_name_jax = 'seaquest', 'Seaquest-MinAtar'


def test_step():
    """ Test a step transition for the env. """
    env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    rng, env_jax = gymnax.make(env_name_jax)

    # Loop over test episodes
    for ep in range(num_episodes):
        obs = env_gym.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            rng, key_step, key_action = jax.random.split(rng, 3)
            state = np_state_to_jax(env_gym, env_name_jax)
            action = env_jax.action_space.sample(key_action)
            action_gym = minatar_action_map(action, env_name_jax)

            reward_gym, done = env_gym.act(action_gym)
            obs_gym = env_gym.state()
            done_gym = env_gym.env.terminal
            obs_jax, state_jax, reward_jax, done_jax, _ = env_jax.step(
                                                                key_step,
                                                                state,
                                                                action)

            # Check correctness of transition
            assert_correct_transit(obs_gym, reward_gym, done_gym,
                                   obs_jax, reward_jax, done_jax,
                                   tolerance)

            # Check that post-transition states are equal
            assert_correct_state(env_gym, env_name_jax, state_jax,
                                 tolerance)

            if done_gym:
                break


def test_reset():
    """ Test reset obs/state is in space of NumPy version. """
    #env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    rng, env_jax = gymnax.make(env_name_jax)
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input)
        # Check state and observation space
        env_jax.state_space.contains(state)
        env_jax.observation_space.contains(obs)


def test_get_obs():
    """ Test observation function. """
    env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    rng, env_jax = gymnax.make(env_name_jax)

    # Loop over test episodes
    for ep in range(num_episodes):
        env_gym.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            rng, key_step, key_action = jax.random.split(rng, 3)
            action = env_jax.action_space.sample(key_action)
            action_gym = minatar_action_map(action, env_name_jax)
            # Step gym environment get state and trafo in jax dict
            reward_gym = env_gym.act(action_gym)
            obs_gym = env_gym.state()
            state = np_state_to_jax(env_gym, env_name_jax)
            obs_jax = env_jax.get_obs(state)
            # Check for correctness of observations
            assert (obs_gym == obs_jax).all()
            done_gym = env_gym.env.terminal
            # Start a new episode if the previous one has terminated
            if done_gym:
                break

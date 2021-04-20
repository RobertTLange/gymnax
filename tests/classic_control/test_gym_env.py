import jax
import jax.numpy as jnp
import gym
import unittest, math
import numpy as np

import gymnax
from gymnax.utils import (np_state_to_jax,
                          assert_correct_transit,
                          assert_correct_state)


num_episodes, num_steps = 10, 150
tolerance = 1e-04


def test_step(env_name):
    """ Test a step transition for the env. """
    env_gym = gym.make(env_name)
    rng, env_jax = gymnax.make(env_name)
    # Create jitted version of step transition function
    jit_step = jax.jit(env_jax.step)

    # Loop over test episodes
    for ep in range(num_episodes):
        obs = env_gym.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            action = env_gym.action_space.sample()
            state = np_state_to_jax(env_gym, env_name)
            obs_gym, reward_gym, done_gym, _ = env_gym.step(action)

            rng, rng_input = jax.random.split(rng)
            obs_jax, state_jax, reward_jax, done_jax, _ = env_jax.step(
                                                                rng_input,
                                                                state,
                                                                action)
            obs_jit, state_jit, reward_jit, done_jit, _ = jit_step(
                                                                rng_input,
                                                                state,
                                                                action)

            # Check correctness of transition
            assert_correct_transit(obs_gym, reward_gym, done_gym,
                                   obs_jax, reward_jax, done_jax,
                                   tolerance)
            assert_correct_transit(obs_gym, reward_gym, done_gym,
                                   obs_jit, reward_jit, done_jit,
                                   tolerance)

            # Check that post-transition states are equal
            assert_correct_state(env_gym, env_name, state_jax,
                                 tolerance)
            assert_correct_state(env_gym, env_name, state_jit,
                                 tolerance)

            if done_gym:
                break


def test_reset(env_name):
    """ Test reset obs/state is in space of OpenAI version. """
    env_gym = gym.make(env_name)
    rng, env_jax = gymnax.make(env_name)
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input)
        # Check state and observation space
        env_jax.state_space.contains(state)
        env_jax.observation_space.contains(obs)

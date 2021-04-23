import jax
import jax.numpy as jnp
import unittest, math
import numpy as np

import gymnax
from gymnax.utils import (np_state_to_jax,
                          assert_correct_transit,
                          assert_correct_state)

from bsuite.environments import (catch, deep_sea, discounting_chain,
                                 memory_chain, umbrella_chain, mnist, bandit)

num_episodes, num_steps = 10, 150
tolerance = 1e-04


def test_step(env_name: str):
    """ Test a step transition for the env. """
    env_bsuite = make_bsuite_env(env_name)
    rng, env_jax = gymnax.make(env_name)
    # Create jitted version of step transition function
    jit_step = jax.jit(env_jax.step)

    # Loop over test episodes
    for ep in range(num_episodes):
        _ = env_bsuite.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            rng, key_action, key_step = jax.random.split(rng, 3)
            action = env_jax.action_space.sample(key_step)
            state = np_state_to_jax(env_bsuite, env_name)
            timestep = env_bsuite.step(action)
            obs_bsuite, reward_bsuite, done_bsuite = (timestep.observation,
                                                      timestep.reward,
                                                      1-timestep.discount)
            obs_jax, state_jax, reward_jax, done_jax, _ = env_jax.step(
                                                                key_step,
                                                                state,
                                                                action)
            obs_jit, state_jit, reward_jit, done_jit, _ = jit_step(
                                                                key_step,
                                                                state,
                                                                action)

            # For umbrella chain ignore reward - randomly sampled!
            if env_name == "UmbrellaChain-bsuite":
                reward_jax = reward_jit = reward_bsuite

            # Check correctness of transition
            assert_correct_transit(obs_bsuite, reward_bsuite, done_bsuite,
                                   obs_jax, reward_jax, done_jax,
                                   tolerance)
            assert_correct_transit(obs_bsuite, reward_bsuite, done_bsuite,
                                   obs_jit, reward_jit, done_jit,
                                   tolerance)

            # Check that post-transition states are equal
            assert_correct_state(env_bsuite, env_name, state_jax,
                                 tolerance)
            assert_correct_state(env_bsuite, env_name, state_jit,
                                 tolerance)

            if done_bsuite:
                break


def test_reset(env_name: str):
    """ Test reset obs/state is in space of OpenAI version. """
    # env_bsuite = make_bsuite_env(env_name)
    rng, env_jax = gymnax.make(env_name)
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input)
        # Check state and observation space
        env_jax.state_space.contains(state)
        env_jax.observation_space.contains(obs)


def make_bsuite_env(env_name: str):
    """ Boilerplate helper for bsuite env generation. """
    if env_name == "Catch-bsuite":
        env = catch.Catch()
    elif env_name == "DeepSea-bsuite":
        env = deep_sea.DeepSea(size=8, randomize_actions=False)
    elif env_name == "DiscountingChain-bsuite":
        env = discounting_chain.DiscountingChain(mapping_seed=0)
    elif env_name == "MemoryChain-bsuite":
        env = memory_chain.MemoryChain(memory_length=5, num_bits=1)
    elif env_name == "UmbrellaChain-bsuite":
        env = umbrella_chain.UmbrellaChain(chain_length=10,
                                           n_distractor=0)
    elif env_name == "MNISTBandit-bsuite":
        env = mnist.MNISTBandit()
    elif env_name == "SimpleBandit-bsuite":
        env = bandit.SimpleBandit()
    return env

"""Tests for bsuite environments."""

import jax
import gymnax
from bsuite.environments import bandit
from bsuite.environments import catch
from bsuite.environments import deep_sea
from bsuite.environments import discounting_chain
from bsuite.environments import memory_chain
from bsuite.environments import mnist
from bsuite.environments import umbrella_chain
from gymnax.utils import state_translate
from gymnax.utils import test_helpers


num_episodes, num_steps, tolerance = 10, 150, 1e-04


def test_step(bsuite_env_name: str):
    """Test a step transition for the env."""
    rng = jax.random.PRNGKey(0)
    env_bsuite = make_bsuite_env(bsuite_env_name)
    env_jax, env_params = gymnax.make(bsuite_env_name)

    # Loop over test episodes
    for _ in range(num_episodes):
        _ = env_bsuite.reset()
        # Loop over test episode steps
        for _ in range(num_steps):
            rng, _, key_step = jax.random.split(rng, 3)
            action = env_jax.action_space(env_params).sample(key_step)
            state = state_translate.np_state_to_jax(
                env_bsuite, bsuite_env_name, get_jax=True
            )
            timestep = env_bsuite.step(action)
            obs_bsuite, reward_bsuite, done_bsuite = (
                timestep.observation,
                timestep.reward,
                1 - timestep.discount,
            )
            obs_jax, state_jax, reward_jax, done_jax, _ = env_jax.step(
                key_step, state, action, env_params
            )
            # For umbrella chain ignore reward - randomly sampled!
            if bsuite_env_name == "UmbrellaChain-bsuite":
                reward_jax = reward_bsuite

            if done_bsuite:
                break
            else:
                # Check that post-transition states are equal
                test_helpers.assert_correct_state(
                    env_bsuite, bsuite_env_name, state_jax, tolerance
                )
                # Check correctness of transition
                test_helpers.assert_correct_transit(
                    obs_bsuite,
                    reward_bsuite,
                    done_bsuite,
                    obs_jax,
                    reward_jax,
                    done_jax,
                    tolerance,
                )


def test_reset(bsuite_env_name: str):
    """Test reset obs/state is in space of OpenAI version."""
    # env_bsuite = make_bsuite_env(env_name)
    rng = jax.random.PRNGKey(0)
    env_jax, env_params = gymnax.make(bsuite_env_name)
    for _ in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input, env_params)
        # Check state and observation space
        env_jax.state_space(env_params).contains(state)
        env_jax.observation_space(env_params).contains(obs)


def make_bsuite_env(bsuite_env_name: str):
    """Boilerplate helper for bsuite env generation."""
    env = None
    if bsuite_env_name == "Catch-bsuite":
        env = catch.Catch()
    elif bsuite_env_name == "DeepSea-bsuite":
        env = deep_sea.DeepSea(size=8, randomize_actions=False)
    elif bsuite_env_name == "DiscountingChain-bsuite":
        env = discounting_chain.DiscountingChain(mapping_seed=0)
    elif bsuite_env_name == "MemoryChain-bsuite":
        env = memory_chain.MemoryChain(num_bits=1, memory_length=5)
    elif bsuite_env_name == "UmbrellaChain-bsuite":
        env = umbrella_chain.UmbrellaChain(n_distractor=0, chain_length=10)
    elif bsuite_env_name == "MNISTBandit-bsuite":
        env = mnist.MNISTBandit()
    elif bsuite_env_name == "SimpleBandit-bsuite":
        env = bandit.SimpleBandit()
    return env

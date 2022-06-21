import jax
import jax.numpy as jnp
import gymnax
from gymnax.utils import (
    np_state_to_jax,
    minatar_action_map,
    assert_correct_transit,
    assert_correct_state,
)

import numpy as np
from minatar.environment import Environment

from gymnax.environments.minatar.freeway import (
    randomize_cars,
    step_agent,
    step_cars,
)
from freeway_helpers import (
    det_randomize_cars_numpy,
    step_cars_numpy,
    step_agent_numpy,
)

num_episodes, num_steps, tolerance = 5, 10, 1e-04
env_name_gym, env_name_jax = "freeway", "Freeway-MinAtar"


def test_step():
    """Test a step transition for the env."""
    rng = jax.random.PRNGKey(0)
    env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    env_jax, env_params = gymnax.make(env_name_jax)

    # Loop over test episodes
    for ep in range(num_episodes):
        obs = env_gym.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            rng, key_step, key_action = jax.random.split(rng, 3)
            state = np_state_to_jax(env_gym, env_name_jax, get_jax=True)
            action = env_jax.action_space(env_params).sample(key_action)
            action_gym = minatar_action_map(action, env_name_jax)

            # Perform step transition for agent & assert correct state dict
            _ = step_agent_numpy(env_gym, action_gym)
            state_jax_post_agent, r, w = step_agent(
                action_gym, state, env_params
            )
            assert_correct_state(
                env_gym, env_name_jax, state_jax_post_agent, tolerance
            )

            # Perform step transition for cars & assert correct state dict
            state_gym_post_agent = np_state_to_jax(
                env_gym, env_name_jax, get_jax=True
            )
            step_cars_numpy(env_gym)
            state_jax_post_cars = step_cars(state_gym_post_agent)
            assert_correct_state(
                env_gym, env_name_jax, state_jax_post_cars, tolerance
            )

            if env_gym.env.terminal:
                break


def test_reset():
    """Test reset obs/state is in space of NumPy version."""
    # env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    rng = jax.random.PRNGKey(0)
    env_jax, env_params = gymnax.make(env_name_jax)
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input, env_params)
        # Check state and observation space
        env_jax.state_space(env_params).contains(state)
        env_jax.observation_space(env_params).contains(obs)


def test_get_obs():
    """Test observation function."""
    rng = jax.random.PRNGKey(0)
    env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
    env_jax, env_params = gymnax.make(env_name_jax)

    # Loop over test episodes
    for ep in range(num_episodes):
        env_gym.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            rng, key_step, key_action = jax.random.split(rng, 3)
            action = env_jax.action_space(env_params).sample(key_action)
            action_gym = minatar_action_map(action, env_name_jax)
            # Step gym environment get state and trafo in jax dict
            _ = env_gym.act(action_gym)
            obs_gym = env_gym.state()
            state = np_state_to_jax(env_gym, env_name_jax, get_jax=True)
            obs_jax = env_jax.get_obs(state)
            # Check for correctness of observations
            assert (obs_gym == obs_jax).all()
            done_gym = env_gym.env.terminal
            # Start a new episode if the previous one has terminated
            if done_gym:
                break


def test_randomize_cars():
    """Test randomization of car locations."""
    # Test initialization version of `randomize_cars`
    for i in range(num_steps):
        speeds = np.random.randint(1, 6, 8)
        directions = np.random.choice([-1, 1], 8)
        cars_gym = det_randomize_cars_numpy(
            speeds, directions, np.zeros((8, 4)), 1
        )
        cars_jax = randomize_cars(
            speeds, directions, jnp.zeros((8, 4), dtype=int), True
        )
        assert (np.array(cars_gym) == cars_jax).all()

    # Test no initialization version of `randomize_cars`
    for i in range(num_steps):
        speeds = np.random.randint(1, 6, 8)
        directions = np.random.choice([-1, 1], 8)
        cars_gym = det_randomize_cars_numpy(
            speeds, directions, np.zeros((8, 4)), False
        )
        cars_jax = randomize_cars(
            speeds, directions, jnp.zeros((8, 4), dtype=int), False
        )
        assert (np.array(cars_gym) == cars_jax).all()
    return

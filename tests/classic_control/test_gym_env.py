import jax
import gym
import gymnax
from gymnax.utils import (
    np_state_to_jax,
    assert_correct_transit,
    assert_correct_state,
)

num_episodes, num_steps, tolerance = 10, 150, 1e-04


def test_step(gym_env_name):
    """Test a step transition for the env."""
    rng = jax.random.PRNGKey(0)
    env_gym = gym.make(gym_env_name)
    env_jax, env_params = gymnax.make(gym_env_name)

    # Loop over test episodes
    for ep in range(num_episodes):
        obs = env_gym.reset()
        # Loop over test episode steps
        for s in range(num_steps):
            action = env_gym.action_space.sample()
            state = np_state_to_jax(env_gym, gym_env_name, get_jax=True)
            obs_gym, reward_gym, done_gym, truncated_gym, _ = env_gym.step(action)

            rng, rng_input = jax.random.split(rng)
            obs_jax, state_jax, reward_jax, done_jax, _ = env_jax.step(
                rng_input, state, action, env_params
            )
            # Check correctness of transition
            assert_correct_transit(
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
                assert_correct_state(
                    env_gym, gym_env_name, state_jax, tolerance
                )
            else:
                break


def test_reset(gym_env_name):
    """Test reset obs/state is in space of OpenAI version."""
    # env_gym = gym.make(env_name)
    rng = jax.random.PRNGKey(0)
    env_jax, env_params = gymnax.make(gym_env_name)
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input, env_params)
        # Check state and observation space
        env_jax.state_space(env_params).contains(state)
        env_jax.observation_space(env_params).contains(obs)

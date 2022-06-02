import jax
import gymnax

num_episodes, num_steps, tolerance = 10, 100, 1e-04


def test_step(misc_env_name: str):
    """Test a step transition for the env."""
    rng = jax.random.PRNGKey(0)
    env_jax, env_params = gymnax.make(misc_env_name)

    # Loop over test episodes
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input, env_params)
        # Loop over test episode steps
        for s in range(num_steps):
            action = env_jax.action_space(env_params).sample(rng_input)
            obs_jax, state_jax, reward_jax, done_jax, _ = env_jax.step(
                rng_input, state, action, env_params
            )
            env_jax.state_space(env_params).contains(state)
            env_jax.observation_space(env_params).contains(obs)


def test_reset(misc_env_name: str):
    """Test reset obs/state is in space of OpenAI version."""
    # env_gym = gym.make(env_name)
    rng = jax.random.PRNGKey(0)
    env_jax, env_params = gymnax.make(misc_env_name)
    for ep in range(num_episodes):
        rng, rng_input = jax.random.split(rng)
        obs, state = env_jax.reset(rng_input, env_params)
        # Check state and observation space
        env_jax.state_space(env_params).contains(state)
        env_jax.observation_space(env_params).contains(obs)

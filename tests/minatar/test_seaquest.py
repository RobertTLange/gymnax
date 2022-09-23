import jax
import gymnax
from gymnax.utils import (
    np_state_to_jax,
    minatar_action_map,
    assert_correct_transit,
    assert_correct_state,
)

from minatar.environment import Environment

from gymnax.environments.minatar.seaquest import (
    step_agent,
    step_bullets,
    step_divers,
    step_e_subs,
    step_e_bullets,
    step_timers,
    surface,
)
from seaquest_helpers import (
    step_agent_numpy,
    step_bullets_numpy,
    step_divers_numpy,
    step_e_subs_numpy,
    step_e_bullets_numpy,
    step_timers_numpy,
    surface_numpy,
)

num_episodes, num_steps, tolerance = 5, 10, 1e-04
env_name_gym, env_name_jax = "seaquest", "Seaquest-MinAtar"


def test_sub_steps():
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
            state = np_state_to_jax(env_gym, env_name_jax)
            action = env_jax.action_space(env_params).sample(key_action)
            action_gym = minatar_action_map(action, env_name_jax)

            step_agent_numpy(env_gym, action_gym)
            state_jax_a = step_agent(state, action_gym, env_params)
            assert_correct_state(env_gym, env_name_jax, state_jax_a, tolerance)

            reward = step_bullets_numpy(env_gym)
            state_jax_b = step_bullets(state_jax_a)
            assert_correct_state(env_gym, env_name_jax, state_jax_a, tolerance)

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
            action = env_jax.action_space.sample(key_action)
            action_gym = minatar_action_map(action, env_name_jax)
            # Step gym environment get state and trafo in jax dict
            reward_gym = env_gym.act(action_gym)
            obs_gym = env_gym.state()
            state = np_state_to_jax(env_gym, env_name_jax)
            obs_jax = env_jax.get_obs(state, env_params)
            # Check for correctness of observations
            assert (obs_gym == obs_jax).all()
            done_gym = env_gym.env.terminal
            # Start a new episode if the previous one has terminated
            if done_gym:
                break

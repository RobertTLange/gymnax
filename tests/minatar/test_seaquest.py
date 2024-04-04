# """Tests for the Seaquest environment."""


# import gymnax
# from gymnax.environments.minatar import seaquest
# from gymnax.tests.minatar import seaquest_helpers
# from gymnax.utils import state_translate
# from gymnax.utils import test_helpers
# import jax
# from minatar import environment


# num_episodes, num_steps, tolerance = 5, 10, 1e-04
# env_name_gym, env_name_jax = "seaquest", "Seaquest-MinAtar"


# def test_sub_steps():
#   """Test a step transition for the env."""
#   rng = jax.random.PRNGKey(0)
#   env_gym = environment.Environment(env_name_gym, sticky_action_prob=0.0)
#   env_jax, env_params = gymnax.make(env_name_jax)


#   # Loop over test episodes
#   for _ in range(num_episodes):
#     _ = env_gym.reset()
#     # Loop over test episode steps
#     for _ in range(num_steps):
#       rng, _, key_action = jax.random.split(rng, 3)
#       state = state_translate.np_state_to_jax(env_gym, env_name_jax)
#       action = env_jax.action_space(env_params).sample(key_action)
#       action_gym = test_helpers.minatar_action_map(action, env_name_jax)


#       seaquest_helpers.step_agent_numpy(env_gym, action_gym)
#       state_jax_a = seaquest.step_agent(state, action_gym, env_params)
#       test_helpers.assert_correct_state(
#           env_gym, env_name_jax, state_jax_a, tolerance
#       )


#       _ = seaquest_helpers.step_bullets_numpy(env_gym)
#       _ = seaquest.step_bullets(state_jax_a)
#       test_helpers.assert_correct_state(
#           env_gym, env_name_jax, state_jax_a, tolerance
#       )


#       if env_gym.env.terminal:
#         break


# def test_reset():
#   """Test reset obs/state is in space of NumPy version."""
#   # env_gym = Environment(env_name_gym, sticky_action_prob=0.0)
#   rng = jax.random.PRNGKey(0)
#   env_jax, env_params = gymnax.make(env_name_jax)
#   for _ in range(num_episodes):
#     rng, rng_input = jax.random.split(rng)
#     obs, state = env_jax.reset(rng_input, env_params)
#     # Check state and observation space
#     env_jax.state_space(env_params).contains(state)
#     env_jax.observation_space(env_params).contains(obs)


# def test_get_obs():
#   """Test observation function."""
#   rng = jax.random.PRNGKey(0)
#   env_gym = environment.Environment(env_name_gym, sticky_action_prob=0.0)
#   env_jax, env_params = gymnax.make(env_name_jax)


#   # Loop over test episodes
#   for _ in range(num_episodes):
#     env_gym.reset()
#     # Loop over test episode steps
#     for _ in range(num_steps):
#       rng, _, key_action = jax.random.split(rng, 3)
#       action = env_jax.action_space.sample(key_action)
#       action_gym = test_helpers.minatar_action_map(action, env_name_jax)
#       # Step gym environment get state and trafo in jax dict
#       _ = env_gym.act(action_gym)
#       obs_gym = env_gym.state()
#       state = state_translate.np_state_to_jax(env_gym, env_name_jax)
#       obs_jax = env_jax.get_obs(state, env_params)
#       # Check for correctness of observations
#       assert (obs_gym == obs_jax).all()
#       done_gym = env_gym.env.terminal
#       # Start a new episode if the previous one has terminated
#       if done_gym:
#         break

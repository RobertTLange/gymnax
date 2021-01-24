import jax
import gym
import gymnax
import unittest, math
import numpy as np


class TestEnv(unittest.TestCase):
    num_episodes, num_steps = 1000, 150
    tolerance = 1e-04
    env_name = 'env-v0'

    def test_acrobot_step(self):
        """ Test a step transition for the env. """
        env = gym.make(TestEnv.env_name)
        rng, reset, step, env_params = gymnax.make(TestEnv.env_name)

        # Loop over test episodes
        for ep in range(TestEnv.num_episodes):
            obs = env.reset()
            # Loop over test episode steps
            for s in range(TestEnv.num_steps):
                action = env.action_space.sample()
                state_gym = env.state[:]
                obs_gym, reward_gym, done_gym, _ = env.step(action)

                rng, rng_input = jax.random.split(rng)
                obs_jax, state_jax, reward_jax, done_jax, _ = step(rng_input,
                                                                   env_params, state_gym,
                                                                   action)

                # Check for correctness of transitions
                assert math.isclose(reward_gym, reward_jax,
                                    rel_tol=TestEnv.tolerance)
                self.assertEqual(done_gym, done_jax)
                assert np.allclose(obs_gym, obs_jax,
                                   atol=TestEnv.tolerance)

    def test_acrobot_reset(self):
        """ Test reset obs/state is in space of OpenAI version. """
        env = gym.make("Acrobot-v1")
        rng, reset, step, env_params = gymnax.make(TestEnv.env_name)
        for ep in range(TestEnv.num_episodes):
            rng, rng_input = jax.random.split(rng)
            obs, state = reset(rng_input, env_params)
            # Check observation space
            for i in range(6):
                self.assertTrue(env.observation_space.low[i]
                                <= obs[i]
                                <= env.observation_space.high[i])

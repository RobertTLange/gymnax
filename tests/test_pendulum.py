import jax
import gym
import gymnax
import unittest, math
import numpy as np


class TestPendulum(unittest.TestCase):
    num_episodes, num_steps = 10, 150
    tolerance = 1e-04
    rng = jax.random.PRNGKey(0)

    def test_pendulum_step(self):
        """ Test a step transition for the env. """
        env = gym.make("Pendulum-v0")
        reset, step, env_params = gymnax.make("Pendulum-v0")

        # Loop over test episodes
        for ep in range(TestPendulum.num_episodes):
            obs = env.reset()
            # Loop over test episode steps
            for s in range(TestPendulum.num_steps):
                action = env.action_space.sample()
                state_gym = env.state[:]
                obs_gym, reward_gym, done_gym, _ = env.step(action)

                TestPendulum.rng, rng_input = jax.random.split(TestPendulum.rng)
                obs_jax, state_jax, reward_jax, done_jax, _ = step(rng_input,
                                                                   env_params, state_gym,
                                                                   action)

                # Check for correctness of transitions
                assert math.isclose(reward_gym, reward_jax,
                                    rel_tol=TestPendulum.tolerance)
                self.assertEqual(done_gym, done_jax)
                assert np.allclose(obs_gym, obs_jax,
                                   atol=TestPendulum.tolerance)

    def test_pendulum_reset(self):
        """ Test reset obs/state is in space of OpenAI version. """
        env = gym.make("Pendulum-v0")
        reset, step, env_params = gymnax.make("Pendulum-v0")
        for ep in range(TestPendulum.num_episodes):
            TestPendulum.rng, rng_input = jax.random.split(TestPendulum.rng)
            obs, state = reset(rng_input)
            # Check observation space
            self.assertTrue(env.observation_space.low[0]
                            <= obs[0]
                            <= env.observation_space.high[0])
            self.assertTrue(env.observation_space.low[1]
                            <= obs[1]
                            <= env.observation_space.high[1])
            self.assertTrue(env.observation_space.low[1]
                            <= obs[1]
                            <= env.observation_space.high[1])
            # Check state space
            self.assertTrue(-np.pi <= state[0] <= np.pi)
            self.assertTrue(-1 <= state[1] <= 1)

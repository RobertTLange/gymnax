import jax
import gym
import gymnax
import unittest, math
import numpy as np
from minatar import Environment


class TestFreeway(unittest.TestCase):
    num_episodes, num_steps = 10, 2
    tolerance = 1e-04
    env_name = 'Freeway-MinAtar'
    action_space = [0, 1, 3]

    def test_freeway_step(self):
        """ Test a step transition for the env. """
        env = Environment('freeway', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)

        # Loop over test episodes
        for ep in range(TestFreeway.num_episodes):
            env.reset()
            # Loop over test episode steps
            for s in range(TestFreeway.num_steps):
                action = np.random.choice(TestFreeway.action_space)
                state_jax = state = {}
                reward_gym, done_gym = env.act(action)
                obs_gym = env.state()

                rng, rng_input = jax.random.split(rng)
                obs_jax, state_jax, reward_jax, done_jax, _ = step(rng_input,
                                                                   env_params, state_jax,
                                                                   action)

                # Check for correctness of transitions
                assert math.isclose(reward_gym, reward_jax,
                                    rel_tol=TestFreeway.tolerance)
                self.assertEqual(done_gym, done_jax)
                assert (obs_gym == obs_jax).all()

                # Start a new episode if the previous one has terminated
                if done_gym:
                    break

    def test_freeway_reset(self):
        """ Test reset obs/state is in space of OpenAI version. """
        env = Environment('breakout', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)
        obs_shape = (10, 10, 6)
        state_keys = []
        for ep in range(TestFreeway.num_episodes):
            rng, rng_input = jax.random.split(rng)
            obs_jax, state = reset(rng_input, env_params)
            # Check existence of state keys
            for k in state_keys:
                assert k in state.keys()
            env.reset()
            obs_gym = env.state()
            # Check observation space
            for i in range(3):
                self.assertTrue(obs_shape[i] == obs_jax.shape[i])
            assert np.allclose(obs_gym, obs_jax,
                               atol=TestFreeway.tolerance)

    def test_freeway_get_obs(self):
        return

    def test_freeway_randomize_cars(self):
        return

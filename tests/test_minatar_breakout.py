import jax
import gym
import gymnax
import unittest, math
import numpy as np
from minatar import Environment


class TestBreakout(unittest.TestCase):
    num_episodes, num_steps = 10, 100
    tolerance = 1e-04
    env_name = 'Breakout-MinAtar'
    action_space = [0, 1, 3]

    def test_breakout_step(self):
        """ Test a step transition for the env. """
        env = Environment('breakout', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestBreakout.env_name)

        # Loop over test episodes
        for ep in range(TestBreakout.num_episodes):
            env.reset()
            # Loop over test episode steps
            for s in range(TestBreakout.num_steps):
                action = np.random.choice(TestBreakout.action_space)
                state_jax = {'ball_dir': env.env.ball_dir,
                             'ball_x': env.env.ball_x,
                             'ball_y': env.env.ball_y,
                             'brick_map': env.env.brick_map,
                             'last_x': env.env.last_x,
                             'last_y': env.env.last_y,
                             'pos': env.env.pos,
                             'strike': env.env.strike,
                             'terminal': env.env.terminal}
                reward_gym, done_gym = env.act(action)
                obs_gym = env.state()

                rng, rng_input = jax.random.split(rng)
                obs_jax, state_jax, reward_jax, done_jax, _ = step(rng_input,
                                                                   env_params, state_jax,
                                                                   action)

                # Check for correctness of transitions
                assert math.isclose(reward_gym, reward_jax,
                                    rel_tol=TestBreakout.tolerance)
                self.assertEqual(done_gym, done_jax)
                assert (obs_gym == obs_jax).all()

                # Start a new episode if the previous one has terminated
                if done_gym:
                    break

    def test_breakout_reset(self):
        """ Test reset obs/state is in space of MinAtar NumPy version. """
        env = Environment('breakout', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestBreakout.env_name)
        obs_shape = (10, 10, 4)
        state_keys = ['ball_dir', 'ball_x', 'ball_y',
                      'brick_map', 'last_x', 'last_y',
                      'pos', 'strike', 'terminal']
        for ep in range(TestBreakout.num_episodes):
            rng, rng_input = jax.random.split(rng)
            obs_jax, state = reset(rng_input, env_params)
            for k in state_keys:
                assert k in state.keys()
            env.reset()
            obs_gym = env.state()
            # Check observation space
            for i in range(3):
                self.assertTrue(obs_shape[i] == obs_jax.shape[i])
            assert np.allclose(obs_gym[:, :, 0], obs_jax[:, :, 0],
                               atol=TestBreakout.tolerance)
            assert np.allclose(obs_gym[:, :, 3], obs_jax[:, :, 3],
                               atol=TestBreakout.tolerance)
            # Last action encoded in separate channel - initially same
            assert np.allclose(obs_jax[:, :, 1], obs_jax[:, :, 2],
                               atol=TestBreakout.tolerance)

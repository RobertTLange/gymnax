import gym
from gymnax import make_env
import unittest, math
import numpy as np


class TestPendulum(unittest.TestCase):
    num_episodes, num_steps = 100, 200
    tolerance = 1e-05

    def test_pendulum_step(self):
        env = gym.make("Pendulum-v0")
        reset, step, env_params = make_env("Pendulum-v0")

        # Loop over test episodes
        for ep in range(TestPendulum.num_episodes):
            obs = env.reset()

            # Loop over test episode steps
            for s in range(TestPendulum.num_steps):
                action = env.action_space.sample()
                state_gym = env.state[:]
                obs_gym, reward_gym, done_gym, _ = env.step(action)
                obs_jax, state_jax, reward_jax, done_jax, _ = step(env_params, state_gym, action)

                # Check for correctness of transitions
                assert math.isclose(reward_gym, reward_jax,
                                    rel_tol=TestPendulum.tolerance)
        return

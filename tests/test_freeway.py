import jax
import gym
import gymnax
import unittest, math
import numpy as np
from minatar import Environment
from gymnax.environments.minatar.freeway import get_obs

class TestFreeway(unittest.TestCase):
    num_episodes, num_steps = 10, 100
    tolerance = 1e-04
    env_name = 'Freeway-MinAtar'
    action_space = [0, 2, 4]

    # def test_freeway_step(self):
    #     """ Test a step transition for the env. """
    #     env = Environment('freeway', sticky_action_prob=0.0)
    #     rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)
    #
    #     # Loop over test episodes
    #     for ep in range(TestFreeway.num_episodes):
    #         env.reset()
    #         # Loop over test episode steps
    #         for s in range(TestFreeway.num_steps):
    #             action = np.random.choice(TestFreeway.action_space)
    #             state_jax = {}
    #             reward_gym, done_gym = env.act(action)
    #             obs_gym = env.state()
    #
    #             rng, rng_input = jax.random.split(rng)
    #             obs_jax, state_jax, reward_jax, done_jax, _ = step(rng_input,
    #                                                                env_params, state_jax,
    #                                                                action)
    #
    #             # Check for correctness of transitions
    #             assert math.isclose(reward_gym, reward_jax,
    #                                 rel_tol=TestFreeway.tolerance)
    #             self.assertEqual(done_gym, done_jax)
    #             assert (obs_gym == obs_jax).all()
    #
    #             # Start a new episode if the previous one has terminated
    #             if done_gym:
    #                 break

    # def test_freeway_reset(self):
    #     """ Test reset obs/state is in space of OpenAI version. """
    #     env = Environment('breakout', sticky_action_prob=0.0)
    #     rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)
    #     obs_shape = (10, 10, 6)
    #     state_keys = []
    #     for ep in range(TestFreeway.num_episodes):
    #         rng, rng_input = jax.random.split(rng)
    #         obs_jax, state = reset(rng_input, env_params)
    #         # Check existence of state keys
    #         for k in state_keys:
    #             assert k in state.keys()
    #         env.reset()
    #         obs_gym = env.state()
    #         # Check observation space
    #         for i in range(3):
    #             self.assertTrue(obs_shape[i] == obs_jax.shape[i])
    #         assert np.allclose(obs_gym, obs_jax,
    #                            atol=TestFreeway.tolerance)

    def test_freeway_get_obs(self):
        env = Environment('freeway', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)

        # Loop over test episodes
        for ep in range(TestFreeway.num_episodes):
            env.reset()
            # Loop over test episode steps
            for s in range(TestFreeway.num_steps):
                action = np.random.choice(TestFreeway.action_space)
                reward_gym, done_gym = env.act(action)
                state_jax = get_jax_state_from_numpy(env)

                # Check for correctness of observations
                obs_gym = get_obs_numpy(env)
                obs_jax = get_obs(state_jax)
                assert (obs_gym == obs_jax).all()

                # Start a new episode if the previous one has terminated
                if done_gym:
                    break

def get_jax_state_from_numpy(env):
    state_jax = {"pos": env.env.pos,
                 "cars": env.env.cars,
                 "move_timer": env.env.move_timer,
                 "terminate_timer": env.env.terminate_timer,
                 "terminal": env.env.terminal}
    return

def get_obs_numpy(env):
    obs = np.zeros((10, 10, len(env.env.channels)), dtype=bool)
    obs[env.env.pos, 4, env.env.channels['chicken']] = 1
    for car in env.env.cars:
        obs[car[1],car[0], env.env.channels['car']] = 1
        back_x = car[0]-1 if car[3]>0 else car[0]+1
        if (back_x < 0):
            back_x=9
        elif (back_x > 9):
            back_x=0
        if (abs(car[3]) == 1):
            trail = env.env.channels['speed1']
        elif (abs(car[3]) == 2):
            trail = env.env.channels['speed2']
        elif (abs(car[3]) == 3):
            trail = env.env.channels['speed3']
        elif (abs(car[3]) == 4):
            trail = env.env.channels['speed4']
        elif (abs(car[3]) == 5):
            trail = env.env.channels['speed5']
        obs[car[1],back_x, trail] = 1
    return obs

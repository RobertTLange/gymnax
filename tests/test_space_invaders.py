import jax
import jax.numpy as jnp
import gymnax
import unittest, math
import numpy as np
from minatar import Environment
from gymnax.environments.minatar.space_invaders import get_obs


class TestSpaceInvaders(unittest.TestCase):
    num_episodes, num_steps = 2, 100
    tolerance = 1e-04
    env_name = 'SpaceInvaders-MinAtar'
    action_space = [0, 1, 3, 5]

    # def test_space_invaders_step(self):
    #     """ Test a step transition for the env. """
    #     env = Environment('space_invaders', sticky_action_prob=0.0)
    #     rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)
    #
    #     # Loop over test episodes
    #     for ep in range(TestFreeway.num_episodes):
    #         env.reset()
    #         # Loop over test episode steps
    #         for s in range(TestFreeway.num_steps):
    #             action = np.random.choice(TestFreeway.action_space)
    #             state_jax = get_jax_state_from_numpy(env)
    #             reward_gym, state_gym_a = step_agent_numpy(env, action)
    #             state_gym_c = step_cars_numpy(env)
    #
    #             state_jax_a, reward_jax, win_cond = step_agent(action,
    #                                                            state_jax,
    #                                                            env_params)
    #             for k in state_jax.keys():
    #                 if type(state_gym_a[k]) == jax.interpreters.xla._DeviceArray:
    #                     assert (state_gym_a[k] == state_jax_a[k]).all()
    #                 else:
    #                     assert state_gym_a[k] == state_jax_a[k]
    #
    #             state_jax_c = step_cars(state_jax_a)
    #             for k in state_jax.keys():
    #                 if type(state_gym_c[k]) == jax.interpreters.xla._DeviceArray:
    #                     assert (state_gym_c[k] == state_jax_c[k]).all()
    #                 else:
    #                     assert state_gym_a[k] == state_jax_c[k]
    #
    #             # Check for correctness of transitions
    #             assert math.isclose(reward_gym, reward_jax,
    #                                 rel_tol=TestFreeway.tolerance)

    def test_space_invaders_reset(self):
        """ Test reset obs/state is in space of OpenAI version. """
        env = Environment('space_invaders', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestSpaceInvaders.env_name)
        obs_shape = (10, 10, 6)
        for ep in range(TestSpaceInvaders.num_episodes):
            rng, rng_input = jax.random.split(rng)
            obs_jax, state_jax = reset(rng_input, env_params)
            state_gym = get_jax_state_from_numpy(env)

            # Check existence and correctness of state keys
            for k in state_gym.keys():
                if type(state_gym[k]) == np.ndarray:
                    assert (state_gym[k] == state_gym[k]).all()
                else:
                    assert state_gym[k] == state_gym[k]
            env.reset()
            obs_gym = env.state()
            # Check observation space
            for i in range(3):
                self.assertTrue(obs_shape[i] == obs_jax.shape[i])
            # Can't check exact obervation due to randomness in cars
            # This is checked independently!

    def test_freeway_get_obs(self):
        env = Environment('space_invaders', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestSpaceInvaders.env_name)

        # Loop over test episodes
        for ep in range(TestSpaceInvaders.num_episodes):
            env.reset()
            # Loop over test episode steps
            for s in range(TestSpaceInvaders.num_steps):
                action = np.random.choice(TestSpaceInvaders.action_space)
                reward_gym, done_gym = env.act(action)
                state_gym = get_jax_state_from_numpy(env)

                # Check for correctness of observations
                obs_gym = get_obs_numpy(env)
                obs_jax = get_obs(state_gym)
                assert (obs_gym == obs_jax).all()

                # Start a new episode if the previous one has terminated
                if done_gym:
                    break


def get_jax_state_from_numpy(env):
    """ A helper for summarizing numpy env info into JAX state. """
    state_jax = {"pos": env.env.pos,
                 "f_bullet_map": env.env.f_bullet_map,
                 "e_bullet_map": env.env.e_bullet_map,
                 "alien_map": env.env.alien_map,
                 "alien_dir": env.env.alien_dir,
                 "enemy_move_interval": env.env.enemy_move_interval,
                 "alien_move_timer": env.env.alien_move_timer,
                 "alien_shot_timer": env.env.alien_shot_timer,
                 "ramp_index": env.env.ramp_index,
                 "shot_timer": env.env.shot_timer,
                 "terminal": env.env.terminal}
    return state_jax


def get_obs_numpy(env):
    """ A helper state(self) function from the numpy env. """
    state = np.zeros((10,10,len(env.env.channels)),dtype=bool)
    state[9,env.env.pos,env.env.channels['cannon']] = 1
    state[:,:, env.env.channels['alien']] = env.env.alien_map
    if(env.env.alien_dir<0):
        state[:,:, env.env.channels['alien_left']] = env.env.alien_map
    else:
        state[:,:, env.env.channels['alien_right']] = env.env.alien_map
    state[:,:, env.env.channels['friendly_bullet']] = env.env.f_bullet_map
    state[:,:, env.env.channels['enemy_bullet']] = env.env.e_bullet_map
    return state

import jax
import jax.numpy as jnp
import gymnax
import unittest, math
import numpy as np
from minatar import Environment
from gymnax.environments.minatar.space_invaders import (get_obs,
                                                        step_agent,
                                                        step_aliens,
                                                        step_shoot,
                                                        get_nearest_alien)


class TestSpaceInvaders(unittest.TestCase):
    num_episodes, num_steps = 2, 100
    tolerance = 1e-04
    env_name = 'SpaceInvaders-MinAtar'
    action_space = [0, 1, 3, 5]

    def test_space_invaders_sub_steps(self):
        """ Test a step transition for the env. """
        env = Environment('space_invaders', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestSpaceInvaders.env_name)

        # Loop over test episodes
        for ep in range(TestSpaceInvaders.num_episodes):
            env.reset()
            # Loop over test episode steps
            for s in range(TestSpaceInvaders.num_steps):
                action = np.random.choice(TestSpaceInvaders.action_space)
                state_jax = get_jax_state_from_numpy(env)
                state_gym_a, term_cond_gym = step_agent_numpy(env, action)
                state_jax_a, term_cond_jax = step_agent(action,
                                                        state_jax,
                                                        env_params)
                for k in state_jax.keys():
                    if type(state_gym_a[k]) == np.ndarray:
                        assert (state_gym_a[k] == state_jax_a[k]).all()
                    else:
                        assert state_gym_a[k] == state_jax_a[k]
                assert term_cond_gym == term_cond_jax

                state_gym_b, term_cond_gym = step_aliens_numpy(env)
                state_jax_b, term_cond_jax = step_aliens(state_jax_a)

                for k in state_jax.keys():
                    if type(state_gym_b[k]) == np.ndarray:
                        assert (state_gym_b[k] == state_jax_b[k]).all()
                    else:
                        assert state_gym_b[k] == state_jax_b[k]
                assert term_cond_gym == term_cond_jax

                state_gym_c, reward_gym = step_shoot_numpy(env)
                state_jax_c, reward_jax = step_shoot(state_jax_b,
                                                     env_params)

                for k in state_jax.keys():
                    if type(state_gym_c[k]) == np.ndarray:
                        assert (state_gym_c[k] == state_jax_c[k]).all()
                    else:
                        assert state_gym_c[k] == state_jax_c[k]
                assert reward_gym == reward_jax

                # Start a new episode if the previous one has terminated
                done_gym = env.env.terminal
                if done_gym:
                    break

    # def test_space_invaders_step(self):
    #     """ Test a step transition for the env. """
    #     env = Environment('space_invaders', sticky_action_prob=0.0)
    #     rng, reset, step, env_params = gymnax.make(TestSpaceInvaders.env_name)
    #
    #     # Loop over test episodes
    #     for ep in range(TestSpaceInvaders.num_episodes):
    #         env.reset()
    #         # Loop over test episode steps
    #         for s in range(TestSpaceInvaders.num_steps):
    #             action = np.random.choice(TestSpaceInvaders.action_space)
    #             state_jax = get_jax_state_from_numpy(env)
    #             obs_jax, state_jax_n, reward_jax, done_jax, info = step(
    #                                     rng, env_params,
    #                                     state_jax, action)
    #             reward_gym, done_gym = env.act(action)
    #             state_gym = get_jax_state_from_numpy(env)
    #
    #             for k in state_jax.keys():
    #                 print(k)
    #                 if type(state_gym[k]) == np.ndarray:
    #                     assert (state_gym[k] == state_jax_n[k]).all()
    #                 else:
    #                     assert state_gym[k] == state_jax_n[k]
    #
    #             # Check the observation
    #             assert (env.state() == obs_jax).all()
    #
    #             # Check the rewards
    #             assert reward_gym == reward_jax
    #
    #             # Start a new episode if the previous one has terminated
    #             done_gym = env.env.terminal
    #             assert done_gym == done_jax
    #             if done_gym:
    #                 break

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
                 "terminal": env.env.terminal,
                 "ramping": True}
    return state_jax


def get_obs_numpy(env):
    """ A helper state(self) function from the numpy env. """
    obs = np.zeros((10,10,len(env.env.channels)),dtype=bool)
    obs[9,env.env.pos,env.env.channels['cannon']] = 1
    obs[:,:, env.env.channels['alien']] = env.env.alien_map
    if(env.env.alien_dir<0):
        obs[:,:, env.env.channels['alien_left']] = env.env.alien_map
    else:
        obs[:,:, env.env.channels['alien_right']] = env.env.alien_map
    obs[:,:, env.env.channels['friendly_bullet']] = env.env.f_bullet_map
    obs[:,:, env.env.channels['enemy_bullet']] = env.env.e_bullet_map
    return obs


def step_agent_numpy(env, action):
    """ Part of numpy env state transition - Only agent. """
    a = env.env.action_map[action]
    shot_cool_down = 5

    # Resolve player action
    if(a=='f' and env.env.shot_timer == 0):
        env.env.f_bullet_map[9,env.env.pos]=1
        env.env.shot_timer = shot_cool_down
    elif(a=='l'):
        env.env.pos = max(0, env.env.pos-1)
    elif(a=='r'):
        env.env.pos = min(9, env.env.pos+1)

    # Update Friendly Bullets
    env.env.f_bullet_map = np.roll(env.env.f_bullet_map, -1, axis=0)
    env.env.f_bullet_map[9,:] = 0

    # Update Enemy Bullets
    env.env.e_bullet_map = np.roll(env.env.e_bullet_map, 1, axis=0)
    env.env.e_bullet_map[0,:] = 0
    return get_jax_state_from_numpy(env), env.env.e_bullet_map[9,env.env.pos]


def step_aliens_numpy(env):
    """ Part of numpy env state transition - Update aliens. """
    terminal_1, terminal_2, terminal_3 = 0, 0, 0
    if(env.env.alien_map[9,env.env.pos]):
        terminal_1 = 1
    if(env.env.alien_move_timer==0):
        env.env.alien_move_timer = min(np.count_nonzero(env.env.alien_map),env.env.enemy_move_interval)
        if((np.sum(env.env.alien_map[:,0])>0 and env.env.alien_dir<0) or (np.sum(env.env.alien_map[:,9])>0 and env.env.alien_dir>0)):
            env.env.alien_dir = -env.env.alien_dir
            if(np.sum(env.env.alien_map[9,:])>0):
                terminal_2 = 1
            env.env.alien_map = np.roll(env.env.alien_map, 1, axis=0)
        else:
            env.env.alien_map = np.roll(env.env.alien_map,
                                        env.env.alien_dir, axis=1)
        if(env.env.alien_map[9,env.env.pos]):
            terminal_3 = 1
    term = (terminal_1 + terminal_2 + terminal_3) > 0
    return get_jax_state_from_numpy(env), term


def step_shoot_numpy(env):
    r = 0
    if(env.env.alien_shot_timer==0):
        env.env.alien_shot_timer = enemy_shot_interval
        nearest_alien = env.env._nearest_alien(env.env.pos)
        env.env.e_bullet_map[nearest_alien[0], nearest_alien[1]] = 1

    kill_locations = np.logical_and(env.env.alien_map,
                                    env.env.alien_map==env.env.f_bullet_map)

    r += np.sum(kill_locations)
    env.env.alien_map[kill_locations] = env.env.f_bullet_map[kill_locations] = 0
    return get_jax_state_from_numpy(env), r


def nearest_alien(env):
    """ Get closest alien to shoot. """
    search_order = [i for i in range(10)]
    search_order.sort(key=lambda x: abs(x-env.env.pos))
    # Loop over distances and check if there is alien left
    for i in search_order:
        if(np.sum(env.env.alien_map[:,i]) > 0):
            return [np.max(np.where(env.env.alien_map[:, i] == 1)), i]
    return None

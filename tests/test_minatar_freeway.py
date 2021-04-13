import jax
import jax.numpy as jnp
import gymnax
import unittest, math
import numpy as np
from minatar import Environment
from gymnax.environments.minatar.freeway import (get_obs, randomize_cars,
                                                 step_cars, step_agent)


class TestFreeway(unittest.TestCase):
    num_episodes, num_steps = 2, 5
    tolerance = 1e-04
    env_name = 'Freeway-MinAtar'
    action_space = [0, 2, 4]

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
                state_jax = get_jax_state_from_numpy(env)
                reward_gym, state_gym_a = step_agent_numpy(env, action)
                state_gym_c = step_cars_numpy(env)

                state_jax_a, reward_jax, win_cond = step_agent(action,
                                                               state_jax,
                                                               env_params)
                for k in state_jax.keys():
                    if type(state_gym_a[k]) == jax.interpreters.xla._DeviceArray:
                        assert (state_gym_a[k] == state_jax_a[k]).all()
                    else:
                        assert state_gym_a[k] == state_jax_a[k]

                state_jax_c = step_cars(state_jax_a)
                for k in state_jax.keys():
                    if type(state_gym_c[k]) == jax.interpreters.xla._DeviceArray:
                        assert (state_gym_c[k] == state_jax_c[k]).all()
                    else:
                        assert state_gym_a[k] == state_jax_c[k]

                # Check for correctness of transitions
                assert math.isclose(reward_gym, reward_jax,
                                    rel_tol=TestFreeway.tolerance)

                # Start a new episode if the previous one has terminated
                done_gym = env.env.terminal
                if done_gym:
                    break

    def test_freeway_reset(self):
        """ Test reset obs/state is in space of OpenAI version. """
        env = Environment('freeway', sticky_action_prob=0.0)
        rng, reset, step, env_params = gymnax.make(TestFreeway.env_name)
        obs_shape = (10, 10, 7)
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
            # Can't check exact obervation due to randomness in cars
            # This is checked independently!

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

    def test_freeway_randomize_cars(self):
        # Test initialization version of `randomize_cars`
        for i in range(TestFreeway.num_steps):
            speeds = np.random.randint(1, 6, 8)
            directions = np.random.choice([-1, 1], 8)
            cars_gym = det_randomize_cars_numpy(speeds, directions,
                                                np.zeros((8, 4)), 1)
            cars_jax = randomize_cars(speeds, directions,
                                      np.zeros((8, 4)), 1)
            assert (np.array(cars_gym) == cars_jax).all()

        # Test no initialization version of `randomize_cars`
        for i in range(TestFreeway.num_steps):
            speeds = np.random.randint(1, 6, 8)
            directions = np.random.choice([-1, 1], 8)
            cars_gym = det_randomize_cars_numpy(speeds, directions,
                                                np.zeros((8, 4)), 1)
            cars_jax = randomize_cars(speeds, directions,
                                      np.zeros((8, 4)), 1)
            assert (np.array(cars_gym) == cars_jax).all()
        return


def get_jax_state_from_numpy(env):
    """ A helper for summarizing numpy env info into JAX state. """
    state_jax = {"pos": env.env.pos,
                 "cars": jnp.array(env.env.cars),
                 "move_timer": env.env.move_timer,
                 "terminate_timer": env.env.terminate_timer,
                 "terminal": env.env.terminal}
    return state_jax


def get_obs_numpy(env):
    """ A helper state(self) function from the numpy env. """
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


def det_randomize_cars_numpy(speeds, directions, old_cars, initialize):
    """ Helper _randomize_cars(self, initialize) function from numpy."""
    # We have extracted all randomness for testing purposes
    speeds_new = directions * speeds
    if(initialize):
        cars = []
        for i in range(8):
            cars += [[0, i+1, abs(speeds_new[i]), speeds_new[i]]]
        return cars
    else:
        for i in range(8):
            old_cars[i][2:4] = [abs(speeds_new[i]), speeds_new[i]]
        return np.array(old_cars)


def step_agent_numpy(env, action):
    """ Helper deterministic part of step transition from numpy."""
    r = 0
    a = env.env.action_map[action]
    if(a=='u' and env.env.move_timer==0):
        env.env.move_timer = 3
        env.env.pos = max(0, env.env.pos-1)
    elif(a=='d' and env.env.move_timer==0):
        env.env.move_timer = 3
        env.env.pos = min(9, env.env.pos+1)
    if(env.env.pos==0):
        r+=1
        env.env.pos = 9
    return r, get_jax_state_from_numpy(env)


def step_cars_numpy(env):
    # Update cars
    for car in env.env.cars:
        if(car[0:2]==[4,env.env.pos]):
            env.env.pos = 9
        if(car[2]==0):
            car[2]=abs(car[3])
            car[0]+=1 if car[3]>0 else -1
            if(car[0]<0):
                car[0]=9
            elif(car[0]>9):
                car[0]=0
            if(car[0:2]==[4,env.env.pos]):
                env.env.pos = 9
        else:
            car[2]-=1

    # Update various timers
    env.env.move_timer-=env.env.move_timer>0
    env.env.terminate_timer-=1
    if(env.env.terminate_timer<0):
        env.env.terminal = True
    return get_jax_state_from_numpy(env)

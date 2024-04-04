"""Visualization of gym environments."""

import gymnasium as gym
import numpy as np


def set_gym_params(gym_env, env_name, params):
    """Set gym environment parameters."""
    if env_name == "Acrobot-v1":
        gym_env.env.LINK_LENGTH_1 = params.link_length_1
        gym_env.env.LINK_LENGTH_2 = params.link_length_2
    elif env_name == "CartPole-v1":
        gym_env.env.x_threshold = params.x_threshold
        gym_env.env.length = params.length
    elif env_name == "Pendulum-v1":
        pass
    elif env_name == "MountainCar-v0":
        gym_env.env.max_position = params.max_position
        gym_env.env.min_position = params.min_position
        gym_env.env.goal_position = params.goal_position
    elif env_name == "MountainCarContinuous-v0":
        gym_env.env.max_position = params.max_position
        gym_env.env.min_position = params.min_position
        gym_env.env.goal_position = params.goal_position
    return


def get_gym_state(state, env_name):
    """Get gym environment state."""
    if env_name == "Acrobot-v1":
        return np.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
            ]
        )
    elif env_name == "CartPole-v1":
        return np.array([state.x, state.x_dot, state.theta, state.theta_dot])
    elif env_name == "Pendulum-v1":
        return np.array([state.theta, state.theta_dot, state.last_u])
    elif env_name == "MountainCar-v0":
        return np.array([state.position, state.velocity])
    elif env_name == "MountainCarContinuous-v0":
        return np.array([state.position, state.velocity])


def init_gym(ax, env, state, params):
    """Initialize gym environment."""
    if env.name == "Pendulum-v1":
        gym_env = gym.make("Pendulum-v0")
    else:
        gym_env = gym.make(env.name)
    gym_env.reset()
    set_gym_params(gym_env, env.name, params)
    gym_state = get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.env.last_u = gym_state[-1]
    gym_env.env.state = gym_state
    rgb_array = gym_env.render(mode="rgb_array")
    ax.set_xticks([])
    ax.set_yticks([])
    gym_env.close()
    return ax.imshow(rgb_array)


def update_gym(im, env, state):
    """Update gym environment."""
    if env.name == "Pendulum-v1":
        gym_env = gym.make("Pendulum-v0")
    else:
        gym_env = gym.make(env.name)
    gym_state = get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.env.last_u = gym_state[-1]
    gym_env.env.state = gym_state
    rgb_array = gym_env.render(mode="rgb_array")
    im.set_data(rgb_array)
    gym_env.close()
    return im

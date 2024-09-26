"""Visualization of gym environments."""

import functools
from typing import Any
import gymnasium as gym
import matplotlib
import matplotlib.axes
import matplotlib.axis
import matplotlib.image
import matplotlib.pyplot
import numpy as np

from gymnax import EnvParams, EnvState
from gymnax.environments.classic_control import (
    acrobot,
    cartpole,
    pendulum,
    mountain_car,
    continuous_mountain_car,
)
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from gymnasium.envs.classic_control import acrobot as gym_acrobot
from gymnasium.envs.classic_control import cartpole as gym_cartpole
from gymnasium.envs.classic_control import pendulum as gym_pendulum
from gymnasium.envs.classic_control import mountain_car as gym_mountain_car
from gymnasium.envs.classic_control import (
    continuous_mountain_car as gym_continuous_mountain_car,
)

from gymnax.wrappers.purerl import GymnaxWrapper

_make_gym_env = functools.lru_cache(maxsize=1)(gym.make)


def _get_gym_env(*args, **kwargs):
    # Create the environment only once.
    env = _make_gym_env(*args, **kwargs)
    # We could also copy it the rest of the time if we want to be extra safe.
    # env = copy.deepcopy(env)
    return env


def init_gym(
    ax: matplotlib.axes.Axes,
    env: Environment[TEnvState, TEnvParams],
    state: TEnvState,
    params: TEnvParams,
):
    """Initialize gym environment."""
    rgb_array = render_env(env, state=state, params=params)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(rgb_array)


def update_gym(
    im: matplotlib.image.AxesImage | Any,
    env: Environment[TEnvState, TEnvParams],
    state: TEnvState,
    params: TEnvParams,
):
    """Update gym environment."""
    rgb_array = render_env(env, state=state, params=params)
    im.set_data(rgb_array)
    return im


def _render_and_close(env: gym.Env):
    with env:
        rgb_array = env.render()
    if not isinstance(rgb_array, np.ndarray):
        raise RuntimeError(
            f"Unable to render gym env {env}: `render()` didnt return an ndarray: {rgb_array}"
        )
    return rgb_array


@functools.singledispatch
def render_env(
    env: Environment[TEnvState, TEnvParams],
    state: TEnvState,
    params: TEnvParams,
) -> np.ndarray:
    """Set gym environment parameters."""
    raise NotImplementedError(
        f"Dont' have a registered handler to render environment {env} with id {env.name}."
    )


@render_env.register(GymnaxWrapper)
def render_wrapped_env(env: GymnaxWrapper, state: EnvState, params: EnvParams):
    # By default set the parameters on the wrapped environment.
    return render_env(env._env, state=state, params=params)


@render_env.register
def render_acrobot(
    env: acrobot.Acrobot,
    params: acrobot.EnvParams,
    state: acrobot.EnvState,
):
    gym_env = _get_gym_env("Acrobot-v1", render_mode="rgb_array").unwrapped
    assert isinstance(gym_env, gym_acrobot.AcrobotEnv)
    gym_env.reset(seed=0)

    # note: These are class attributes on the gym env, but setting them here is fine since they are
    # accessed with self.<ATTR> in the gym env.
    gym_env.AVAIL_TORQUE = params.available_torque.tolist()  # type: ignore
    gym_env.dt = params.dt
    gym_env.LINK_LENGTH_1 = params.link_length_1
    gym_env.LINK_LENGTH_2 = params.link_length_2
    gym_env.LINK_MASS_1 = params.link_mass_1
    gym_env.LINK_MASS_2 = params.link_mass_2
    gym_env.LINK_COM_POS_1 = params.link_com_pos_1
    gym_env.LINK_COM_POS_2 = params.link_com_pos_2
    gym_env.LINK_MOI = params.link_moi
    gym_env.MAX_VEL_1 = params.max_vel_1
    gym_env.MAX_VEL_2 = params.max_vel_2
    gym_env.torque_noise_max = params.torque_noise_max
    # gym_env. = params.max_steps_in_episode

    gym_env.state = np.array(
        [
            state.joint_angle1,
            state.joint_angle2,
            state.velocity_1,
            state.velocity_2,
        ]
    )
    return _render_and_close(gym_env)


@render_env.register
def get_gym_cartpole_env_with_same_state(
    env: cartpole.CartPole, state: cartpole.EnvState, params: cartpole.EnvParams
):
    gym_env = _get_gym_env("CartPole-v1", render_mode="rgb_array").unwrapped
    assert isinstance(gym_env, gym_cartpole.CartPoleEnv)

    gym_env.gravity = params.gravity
    gym_env.masscart = params.masscart
    gym_env.masspole = params.masspole
    gym_env.total_mass = params.total_mass
    gym_env.length = params.length  # actually half the pole's length
    gym_env.polemass_length = params.polemass_length
    gym_env.force_mag = params.force_mag
    gym_env.tau = params.tau  # seconds between state updates

    gym_env.x_threshold = params.x_threshold
    gym_env.theta_threshold_radians = params.theta_threshold_radians

    gym_env.reset(seed=0)

    gym_env.state = np.array([state.x, state.x_dot, state.theta, state.theta_dot])
    return _render_and_close(gym_env)


@render_env.register
def render_pendulum(
    env: pendulum.Pendulum, state: pendulum.EnvState, params: pendulum.EnvParams
):
    gym_env = _get_gym_env("Pendulum-v1", g=params.g, render_mode="rgb_array").unwrapped
    assert isinstance(gym_env, gym_pendulum.PendulumEnv)
    gym_env.max_speed = params.max_speed  # type: ignore
    gym_env.max_torque = params.max_torque
    gym_env.dt = params.dt
    gym_env.g = params.g
    gym_env.m = params.m
    gym_env.l = params.l
    gym_env.reset(seed=0)
    gym_env.state = np.array([state.theta, state.theta_dot, state.last_u])
    gym_env.last_u = state.last_u.item()
    return _render_and_close(gym_env)


@render_env.register
def render_mountain_car(
    env: mountain_car.MountainCar,
    state: mountain_car.EnvState,
    params: mountain_car.EnvParams,
):
    gym_env = _get_gym_env("MountainCar-v0", render_mode="rgb_array").unwrapped
    assert isinstance(gym_env, gym_mountain_car.MountainCarEnv)

    gym_env.max_position = params.max_position
    gym_env.min_position = params.min_position
    gym_env.goal_position = params.goal_position

    gym_env.min_position = params.min_position
    gym_env.max_position = params.max_position
    gym_env.max_speed = params.max_speed
    gym_env.goal_position = params.goal_position
    gym_env.goal_velocity = params.goal_velocity  # type: ignore

    gym_env.force = params.force
    gym_env.gravity = params.gravity

    gym_env.low = np.array([params.min_position, -params.max_speed], dtype=np.float32)
    gym_env.high = np.array([params.max_position, params.max_speed], dtype=np.float32)

    gym_env.reset(seed=0)
    gym_env.state = np.array([state.position, state.velocity])
    return _render_and_close(gym_env)


@render_env.register
def render_mountain_car_continuous(
    env: continuous_mountain_car.ContinuousMountainCar,
    state: continuous_mountain_car.EnvState,
    params: continuous_mountain_car.EnvParams,
):
    gym_env = _get_gym_env(
        "MountainCarContinuous-v0", render_mode="rgb_array"
    ).unwrapped
    assert isinstance(gym_env, gym_continuous_mountain_car.Continuous_MountainCarEnv)

    gym_env.min_action = params.min_action
    gym_env.max_action = params.max_action
    gym_env.min_position = params.min_position
    gym_env.max_position = params.max_position
    gym_env.max_speed = params.max_speed
    gym_env.goal_position = params.goal_position
    gym_env.goal_velocity = params.goal_velocity  # type: ignore
    gym_env.power = params.power

    gym_env.low_state = np.array(
        [params.min_position, -params.max_speed], dtype=np.float32
    )
    gym_env.high_state = np.array(
        [params.max_position, params.max_speed], dtype=np.float32
    )

    gym_env.reset(seed=0)

    gym_env.state = np.array([state.position, state.velocity])
    return _render_and_close(gym_env)

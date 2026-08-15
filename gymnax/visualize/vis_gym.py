"""Visualization adapters for Gymnasium classic-control environments."""

from typing import Any

import gymnasium as gym
import numpy as np

SUPPORTED_GYM_ENVS = (
    "Acrobot-v1",
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
)

_PARAMETER_MAPPINGS = {
    "Acrobot-v1": {
        "LINK_LENGTH_1": "link_length_1",
        "LINK_LENGTH_2": "link_length_2",
        "LINK_MASS_1": "link_mass_1",
        "LINK_MASS_2": "link_mass_2",
        "LINK_COM_POS_1": "link_com_pos_1",
        "LINK_COM_POS_2": "link_com_pos_2",
        "LINK_MOI": "link_moi",
        "dt": "dt",
        "MAX_VEL_1": "max_vel_1",
        "MAX_VEL_2": "max_vel_2",
        "torque_noise_max": "torque_noise_max",
    },
    "CartPole-v1": {
        "gravity": "gravity",
        "masscart": "masscart",
        "masspole": "masspole",
        "total_mass": "total_mass",
        "length": "length",
        "polemass_length": "polemass_length",
        "force_mag": "force_mag",
        "tau": "tau",
        "theta_threshold_radians": "theta_threshold_radians",
        "x_threshold": "x_threshold",
    },
    "Pendulum-v1": {
        "max_speed": "max_speed",
        "max_torque": "max_torque",
        "dt": "dt",
        "g": "g",
        "m": "m",
        "l": "l",
    },
    "MountainCar-v0": {
        "min_position": "min_position",
        "max_position": "max_position",
        "max_speed": "max_speed",
        "goal_position": "goal_position",
        "goal_velocity": "goal_velocity",
        "force": "force",
        "gravity": "gravity",
    },
    "MountainCarContinuous-v0": {
        "min_action": "min_action",
        "max_action": "max_action",
        "min_position": "min_position",
        "max_position": "max_position",
        "max_speed": "max_speed",
        "goal_position": "goal_position",
        "goal_velocity": "goal_velocity",
        "power": "power",
    },
}


def _as_scalar(value: Any) -> float:
    return float(np.asarray(value))


def set_gym_params(gym_env: Any, env_name: str, params: Any) -> None:
    """Copy render-relevant Gymnax parameters to an unwrapped Gymnasium env."""
    for gym_attr, gymnax_attr in _PARAMETER_MAPPINGS[env_name].items():
        setattr(gym_env, gym_attr, _as_scalar(getattr(params, gymnax_attr)))


def get_gym_state(state: Any, env_name: str) -> np.ndarray:
    """Convert a Gymnax state into the matching Gymnasium state vector."""
    values: tuple[Any, ...]
    if env_name == "Acrobot-v1":
        values = (
            state.joint_angle1,
            state.joint_angle2,
            state.velocity_1,
            state.velocity_2,
        )
    elif env_name == "CartPole-v1":
        values = (state.x, state.x_dot, state.theta, state.theta_dot)
    elif env_name == "Pendulum-v1":
        values = (state.theta, state.theta_dot)
    elif env_name in ("MountainCar-v0", "MountainCarContinuous-v0"):
        values = (state.position, state.velocity)
    else:
        raise ValueError(f"Unsupported Gymnasium visualizer: {env_name}")
    return np.asarray(values, dtype=np.float64)


def render_gym_frame(env_name: str, state: Any, params: Any) -> np.ndarray:
    """Render one Gymnax state with the modern Gymnasium rgb-array API."""
    if env_name not in SUPPORTED_GYM_ENVS:
        raise ValueError(f"Unsupported Gymnasium visualizer: {env_name}")

    gym_env = gym.make(env_name, render_mode="rgb_array")
    try:
        gym_env.reset()
        unwrapped_env: Any = gym_env.unwrapped
        set_gym_params(unwrapped_env, env_name, params)
        unwrapped_env.state = get_gym_state(state, env_name)
        if env_name == "Pendulum-v1":
            unwrapped_env.last_u = np.asarray(state.last_u, dtype=np.float64)
        frame: Any = gym_env.render()
    finally:
        gym_env.close()

    if not isinstance(frame, np.ndarray):
        raise RuntimeError("Gymnasium renderer did not return an RGB array")
    return frame


def init_gym(ax: Any, env: Any, state: Any, params: Any) -> Any:
    """Initialize a classic-control animation frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(render_gym_frame(env.name, state, params))


def update_gym(im: Any, env: Any, state: Any, params: Any) -> Any:
    """Update a classic-control animation frame."""
    im.set_data(render_gym_frame(env.name, state, params))
    return im

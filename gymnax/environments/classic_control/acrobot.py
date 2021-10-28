import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class Acrobot(environment.Environment):
    """
    JAX Compatible version of Acrobot-v1 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
    Note that we only implement the default 'book' version.
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "dt": 0.2,
            "link_length_1": 1.0,
            "link_length_2": 1.0,
            "link_mass_1": 1.0,
            "link_mass_2": 1.0,
            "link_com_pos_1": 0.5,
            "link_com_pos_2": 0.5,
            "link_moi": 1.0,
            "max_vel_1": 4 * jnp.pi,
            "max_vel_2": 9 * jnp.pi,
            "available_torque": jnp.array([-1.0, 0.0, +1.0]),
            "torque_noise_max": 0.0,
            "max_steps_in_episode": 500,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        torque = params["available_torque"][action]
        # Add noise to force action - always sample - conditionals in JAX
        torque = torque + jax.random.uniform(
            key,
            shape=(),
            minval=-params["torque_noise_max"],
            maxval=params["torque_noise_max"],
        )

        # Augment state with force action so it can be passed to ds/dt
        s_augmented = jnp.array(
            [
                state["joint_angle1"],
                state["joint_angle2"],
                state["velocity_1"],
                state["velocity_2"],
                torque,
            ]
        )
        ns = rk4(s_augmented, params)
        joint_angle1 = wrap(ns[0], -jnp.pi, jnp.pi)
        joint_angle2 = wrap(ns[1], -jnp.pi, jnp.pi)
        velocity_1 = jnp.clip(ns[2], -params["max_vel_1"], params["max_vel_1"])
        velocity_2 = jnp.clip(ns[3], -params["max_vel_2"], params["max_vel_2"])

        done_angle = -jnp.cos(joint_angle1) - jnp.cos(joint_angle2 + joint_angle1) > 1.0
        reward = -1.0 * (1 - done_angle)

        # Update state dict and evaluate termination conditions
        state = {
            "joint_angle1": joint_angle1,
            "joint_angle2": joint_angle2,
            "velocity_1": velocity_1,
            "velocity_2": velocity_2,
            "time": state["time"] + 1,
        }
        done = self.is_terminal(state, params)
        state["terminal"] = done
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(4,), minval=-0.1, maxval=0.1)
        state = {
            "joint_angle1": init_state[0],
            "joint_angle2": init_state[1],
            "velocity_1": init_state[2],
            "velocity_2": init_state[3],
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        return jnp.array(
            [
                jnp.cos(state["joint_angle1"]),
                jnp.sin(state["joint_angle1"]),
                jnp.cos(state["joint_angle2"]),
                jnp.sin(state["joint_angle2"]),
                state["velocity_1"],
                state["velocity_2"],
            ]
        )

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        # Check termination and construct updated state
        done_angle = (
            -jnp.cos(state["joint_angle1"])
            - jnp.cos(state["joint_angle2"] + state["joint_angle1"])
            > 1.0
        )
        # Check number of steps in episode termination condition
        done_steps = state["time"] > params["max_steps_in_episode"]
        done = jnp.logical_or(done_angle, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Acrobot-v1"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        high = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                params["max_vel_1"],
                params["max_vel_2"],
            ],
            dtype=jnp.float32,
        )
        return spaces.Box(-high, high, (6,), jnp.float32)

    def state_space(self, params: dict):
        """State space of the environment."""
        high = jnp.array(
            [
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                params["max_vel_1"],
                params["max_vel_2"],
            ],
            dtype=jnp.float32,
        )
        return spaces.Dict(
            {
                "joint_angle1": spaces.Box(-high[0], high[0], (), jnp.float32),
                "joint_angle2": spaces.Box(-high[1], high[1], (), jnp.float32),
                "velocity_1": spaces.Box(-high[2], high[2], (), jnp.float32),
                "velocity_2": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )


def dsdt(s_augmented, t, params):
    """Compute time derivative of the state change - Use for ODE int."""
    m1, m2 = params["link_mass_1"], params["link_mass_2"]
    l1 = params["link_length_1"]
    lc1, lc2 = params["link_com_pos_1"], params["link_com_pos_2"]
    I1, I2 = params["link_moi"], params["link_moi"]
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1, theta2, dtheta1, dtheta2 = s
    d1 = (
        m1 * lc1 ** 2
        + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2
    ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])


def wrap(x, m, M):
    """For example, m = -180, M = 180 (degrees), x = 360 --> returns 0."""
    diff = M - m
    diff_x_M = x - M
    diff_x_m = x - m
    go_down = diff_x_M > 0
    go_up = diff_x_m < 0
    how_often = jnp.ceil(diff_x_M / diff) * go_down + jnp.ceil(diff_x_m / diff) * go_up
    x_out = x - how_often * diff * go_down + how_often * diff * go_up
    return x_out


def rk4(y0, params):
    """Runge-Kutta integration of ODE - Difference to OpenAI: Only 1 step!"""
    dt2 = params["dt"] / 2.0
    k1 = dsdt(y0, 0, params)
    k2 = dsdt(y0 + dt2 * k1, dt2, params)
    k3 = dsdt(y0 + dt2 * k2, dt2, params)
    k4 = dsdt(y0 + params["dt"] * k3, params["dt"], params)
    yout = y0 + params["dt"] / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout

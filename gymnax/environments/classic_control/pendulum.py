import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class Pendulum(environment.Environment):
    """
    JAX Compatible version of Pendulum-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self):
        """Default environment parameters for Pendulum-v0."""
        return {
            "max_speed": 8,
            "max_torque": 2.0,
            "dt": 0.05,
            "g": 10.0,
            "m": 1.0,
            "l": 1.0,
            "max_steps_in_episode": 200,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: float, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Integrate pendulum ODE and return transition."""
        u = jnp.clip(action, -params["max_torque"], params["max_torque"])
        reward = -(
            angle_normalize(state["theta"]) ** 2
            + 0.1 * state["theta_dot"] ** 2
            + 0.001 * (u ** 2)
        )

        newthdot = state["theta_dot"] + (
            (
                3 * params["g"] / (2 * params["l"]) * jnp.sin(state["theta"])
                + 3.0 / (params["m"] * params["l"] ** 2) * u
            )
            * params["dt"]
        )

        newthdot = jnp.clip(newthdot, -params["max_speed"], params["max_speed"])
        newth = state["theta"] + newthdot * params["dt"]

        # Update state dict and evaluate termination conditions
        state = {
            "theta": newth.squeeze(),
            "theta_dot": newthdot.squeeze(),
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

    def reset_env(self, key: PRNGKey, params: dict):
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        timestep = 0
        state = {
            "theta": state[0],
            "theta_dot": state[1],
            "time": timestep,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state):
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [jnp.cos(state["theta"]), jnp.sin(state["theta"]), state["theta_dot"]]
        ).squeeze()

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state["time"] > params["max_steps_in_episode"]
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v1"

    def action_space(self, params: dict):
        """Action space of the environment."""
        return spaces.Box(
            low=-params["max_torque"],
            high=params["max_torque"],
            shape=(),
            dtype=jnp.float32,
        )

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        high = jnp.array([1.0, 1.0, params["max_speed"]], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(3,), dtype=jnp.float32)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "theta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )


def angle_normalize(x):
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
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
        # Default environment parameters for Pendulum-v0
        self.env_params = FrozenDict({"max_speed": 8,
                                      "max_torque": 2.,
                                      "dt": 0.05,
                                      "g": 10.0,
                                      "m": 1.,
                                      "l": 1.,
                                      "max_steps_in_episode": 200})


    def step(self, key: PRNGKey, state: dict, action: float
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Integrate pendulum ODE and return transition. """
        u = jnp.clip(action, -self.env_params["max_torque"],
                     self.env_params["max_torque"])
        reward = - (angle_normalize(state["theta"]) ** 2
                    + .1 * state["theta_dot"] ** 2 + .001 * (u ** 2))

        newthdot = state["theta_dot"] + ((-3 * self.env_params["g"] /
                            (2 * self.env_params["l"]) * jnp.sin(state["theta"]
                             + jnp.pi) + 3. /
                            (self.env_params["m"] *
                             self.env_params["l"] ** 2) * u)
                                         * self.env_params["dt"])
        newth = state["theta"] + newthdot * self.env_params["dt"]
        newthdot = jnp.clip(newthdot, -self.env_params["max_speed"],
                            self.env_params["max_speed"])

        # Update state dict and evaluate termination conditions
        state = {"theta": newth.squeeze(),
                 "theta_dot": newthdot.squeeze(),
                 "time": state["time"] + 1}
        done = self.is_terminal(state)
        state["terminal"] = done
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done,
                {"discount": self.discount(state)})

    def reset(self, key: PRNGKey):
        """ Reset environment state by sampling theta, theta_dot. """
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(key, shape=(2,),
                                   minval=-high, maxval=high)
        timestep = 0
        state = {"theta": state[0],
                 "theta_dot": state[1],
                 "time": timestep,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state):
        """ Return angle in polar coordinates and change. """
        return jnp.array([jnp.cos(state["theta"]),
                          jnp.sin(state["theta"]),
                          state["theta_dot"]]).squeeze()

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        # Check number of steps in episode termination condition
        done = (state["time"] > self.env_params["max_steps_in_episode"])
        return done

    @property
    def name(self) -> str:
        """ Environment name. """
        return "Pendulum-v0"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Box(low=-self.env_params["max_torque"],
                          high=self.env_params["max_torque"],
                          shape=(), dtype=jnp.float32)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        high = jnp.array([1., 1., self.env_params["max_speed"]],
                         dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(3,), dtype=jnp.float32)

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
           {"theta": spaces.Box(-jnp.finfo(jnp.float32).max,
                                jnp.finfo(jnp.float32).max, (), jnp.float32),
            "theta_dot": spaces.Box(-jnp.finfo(jnp.float32).max,
                                    jnp.finfo(jnp.float32).max, (),
                                    jnp.float32),
            "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
            "terminal": spaces.Discrete(2)})


def angle_normalize(x):
    """ Normalize the angle - radians. """
    return (((x+jnp.pi) % (2*jnp.pi)) - jnp.pi)

import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey


class ContinuousMountainCar(environment.Environment):
    """
    JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
    """
    def __init__(self):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"min_action": -1.0,
                                      "max_action": 1,
                                      "min_position": -1.2,
                                      "max_position": 0.6,
                                      "max_speed": 0.07,
                                      "goal_position": 0.45,
                                      "goal_velocity": 0.0,
                                      "power": 0.0015,
                                      "gravity": 0.0025,
                                      "max_steps_in_episode": 999})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        force = jnp.clip(action, self.env_params["min_action"], self.env_params["max_action"])
        velocity = (state["velocity"] + force * self.env_params["power"]
                    - jnp.cos(3 * state["position"]) *
                    self.env_params["gravity"])
        velocity = jnp.clip(velocity, -self.env_params["max_speed"],
                            self.env_params["max_speed"])
        position = state["position"] + velocity
        position = jnp.clip(position, self.env_params["min_position"],
                            self.env_params["max_position"])
        velocity = velocity * (1 - (position >=
                                    self.env_params["goal_position"])
                               * (velocity < 0))

        reward = (-0.1*action**2
                  + 100*((position >= self.env_params["goal_position"])
                         * (velocity >= self.env_params["goal_velocity"])))

        # Update state dict and evaluate termination conditions
        state = {"position": position,
                 "velocity": velocity,
                 "time": state["time"] + 1}
        done = self.is_terminal(state)
        state["terminal"] = done
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done,
                {"discount": self.discount(state)})

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        init_state = jax.random.uniform(key, shape=(),
                                        minval=-0.6, maxval=-0.4)
        state = {"position": init_state,
                 "velocity": 0,
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        return jnp.array([state["position"],
                          state["velocity"]]).squeeze()

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done1 = ((state["position"] >= self.env_params["goal_position"])
                 * (state["velocity"] >= self.env_params["goal_velocity"]))
        # Check number of steps in episode termination condition
        done_steps = (state["time"] >
                      self.env_params["max_steps_in_episode"])
        done = jnp.logical_or(done1, done_steps)
        return done.squeeze()

    @property
    def name(self) -> str:
        """ Environment name. """
        return "ContinuousMountainCar-v0"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Continuous(minval=self.env_params["min_action"],
                                 maxval=self.env_params["max_action"])

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        low = jnp.array([self.env_params["min_position"],
                         -self.env_params["max_speed"]],
                        dtype=jnp.float32)
        high = jnp.array([self.env_params["max_position"],
                          self.env_params["max_speed"]],
                         dtype=jnp.float32)
        return spaces.Box(low, high, (2,))

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(["position", "velocity", "time", "terminal"])

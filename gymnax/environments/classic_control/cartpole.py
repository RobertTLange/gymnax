import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict
from gymnax.environments import environment
from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey

# JAX Compatible version of CartPole-v0 OpenAI gym environment. Source:
# github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py



class CartPole(environment.Environment):
    def __init__(self):
        super().__init__()
        # Default environment parameters for CartPole-v0
        self.env_params = FrozenDict({"gravity": 9.8,
                                      "masscart": 1.0,
                                      "masspole": 0.1,
                                      "total_mass": 1.0 + 0.1,  # (masscart + masspole)
                                      "length": 0.5,
                                      "polemass_length": 0.05,  # (masspole * length)
                                      "force_mag": 10.0,
                                      "tau": 0.02,
                                      "theta_threshold_radians": 12*2*jnp.pi/360,
                                      "x_threshold": 2.4,
                                      "max_steps_in_episode": 200})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict,float, bool, dict]:
        """ Performs step transitions in the environment."""
        force = (self.env_params["force_mag"] * action
                 - self.env_params["force_mag"]*(1-action))
        costheta = jnp.cos(state["theta"])
        sintheta = jnp.sin(state["theta"])

        temp = (force + self.env_params["polemass_length"]
                * state["theta_dot"] ** 2
                * sintheta) / self.env_params["total_mass"]
        thetaacc = ((self.env_params["gravity"] * sintheta - costheta * temp) /
                    (self.env_params["length"] * (4.0 / 3.0 -
                                                  self.env_params["masspole"]
                     * costheta ** 2 / self.env_params["total_mass"])))
        xacc = (temp - self.env_params["polemass_length"] * thetaacc * costheta
                / self.env_params["total_mass"])

        # Only default Euler integration option available here!
        x = state["x"] + self.env_params["tau"] * state["x_dot"]
        x_dot = state["x_dot"] + self.env_params["tau"] * xacc
        theta = state["theta"] + self.env_params["tau"] * state["theta_dot"]
        theta_dot = state["theta_dot"] + self.env_params["tau"] * thetaacc

        # Check termination criteria
        done1 = jnp.logical_or(x < -self.env_params["x_threshold"],
                               x > self.env_params["x_threshold"])
        done2 = jnp.logical_or(theta < -self.env_params["theta_threshold_radians"],
                               theta > self.env_params["theta_threshold_radians"])

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - state["terminal"]

        # Check number of steps in episode termination condition
        done_steps = (state["time"] + 1 > self.env_params["max_steps_in_episode"])
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        state = {"x": x,
                 "x_dot": x_dot,
                 "theta": theta,
                 "theta_dot": theta_dot,
                 "time": state["time"] + 1,
                 "terminal": done}
        return self.get_obs(state), state, reward, done, {}

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05,
                                        maxval=0.05, shape=(4,))
        timestep = 0
        state = {"x": init_state[0],
                 "x_dot": init_state[1],
                 "theta": init_state[2],
                 "theta_dot": init_state[3],
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Applies observation function to state."""
        return jnp.array([state["x"], state["x_dot"],
                          state["theta"], state["theta_dot"]])

    @property
    def name(self) -> str:
        """Distribution name."""
        return "CartPole-v0"

    @property
    def action_space(self):
        """ Action space of the environment."""

    @property
    def observation_space(self):
        """ Action space of the environment."""

    @property
    def state_space(self):
        """ Action space of the environment."""

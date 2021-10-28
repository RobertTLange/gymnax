import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
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

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "min_action": -1.0,
            "max_action": 1,
            "min_position": -1.2,
            "max_position": 0.6,
            "max_speed": 0.07,
            "goal_position": 0.45,
            "goal_velocity": 0.0,
            "power": 0.0015,
            "gravity": 0.0025,
            "max_steps_in_episode": 999,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: float, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        force = jnp.clip(action, params["min_action"], params["max_action"])
        velocity = (
            state["velocity"]
            + force * params["power"]
            - jnp.cos(3 * state["position"]) * params["gravity"]
        )
        velocity = jnp.clip(velocity, -params["max_speed"], params["max_speed"])
        position = state["position"] + velocity
        position = jnp.clip(position, params["min_position"], params["max_position"])
        velocity = velocity * (
            1 - (position >= params["goal_position"]) * (velocity < 0)
        )

        reward = -0.1 * action ** 2 + 100 * (
            (position >= params["goal_position"])
            * (velocity >= params["goal_velocity"])
        )

        # Update state dict and evaluate termination conditions
        state = {
            "position": position.squeeze(),
            "velocity": velocity.squeeze(),
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
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = {"position": init_state, "velocity": 0.0, "time": 0, "terminal": False}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        return jnp.array([state["position"], state["velocity"]]).squeeze()

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done1 = (state["position"] >= params["goal_position"]) * (
            state["velocity"] >= params["goal_velocity"]
        )
        # Check number of steps in episode termination condition
        done_steps = state["time"] > params["max_steps_in_episode"]
        done = jnp.logical_or(done1, done_steps)
        return done.squeeze()

    @property
    def name(self) -> str:
        """Environment name."""
        return "ContinuousMountainCar-v0"

    @property
    def action_space(self, params: dict):
        """Action space of the environment."""
        return spaces.Box(
            low=params["min_action"],
            high=params["max_action"],
            shape=(),
        )

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        low = jnp.array(
            [params["min_position"], -params["max_speed"]],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params["max_position"], params["max_speed"]],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, shape=(2,), dtype=jnp.float32)

    def state_space(self, params: dict):
        """State space of the environment."""
        low = jnp.array(
            [params["min_position"], -params["max_speed"]],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params["max_position"], params["max_speed"]],
            dtype=jnp.float32,
        )
        return spaces.Dict(
            {
                "position": spaces.Box(low[0], high[0], (), dtype=jnp.float32),
                "velocity": spaces.Box(low[1], high[1], (), dtype=jnp.float32),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

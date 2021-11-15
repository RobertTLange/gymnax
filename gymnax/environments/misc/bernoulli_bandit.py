import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class BernoulliBandit(environment.Environment):
    """
    JAX Compatible version of a Bernoulli bandit environment as in Wang et al.
    2017
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self):
        # Default environment parameters
        return {"reward_prob": 0.1, "max_steps_in_episode": 100}

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Sample bernoulli reward, increase counter, construct input."""
        reward = jax.random.bernoulli(key, state["reward_probs"][action]).astype(
            jnp.int32
        )

        # Update state dict and evaluate termination conditions
        state = {
            "last_action": action,
            "last_reward": reward,
            "reward_probs": state["reward_probs"],
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
        # Sample reward function + construct state as concat with timestamp
        p1 = jax.random.choice(
            key,
            jnp.array([params["reward_prob"], 1 - params["reward_prob"]]),
            shape=(1,),
        ).squeeze()

        state = {
            "last_action": 0,
            "last_reward": 0,
            "reward_probs": jnp.array([p1, 1 - p1]),
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict):
        """Concatenate reward, one-hot action and time stamp."""
        action_one_hot = jax.nn.one_hot(state["last_action"], 2).squeeze()
        return jnp.hstack([state["last_reward"], action_one_hot, state["time"]])

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state["time"] > params["max_steps_in_episode"]
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "BernoulliBandit-misc"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        low = jnp.array(
            [0, 0, 0, 0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [1, 1, 1, params["max_steps_in_episode"]],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (4,), jnp.float32)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(2),
                "last_reward": spaces.Discrete(2),
                "reward_probs": spaces.Box(0, 1, (2,), jnp.float32),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

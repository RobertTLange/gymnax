import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class DiscountingChain(environment.Environment):
    """
    JAX Compatible version of DiscountingChain bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/discounting_chain.py
    """

    def __init__(self, n_actions: int = 5, mapping_seed: int = 0):
        super().__init__()
        self.n_actions = n_actions
        self.mapping_seed = mapping_seed

        # Setup reward fct fron mapping seed - random sampling outside of env
        reward = jnp.ones(self.n_actions)
        self.reward = jax.ops.index_update(
            reward, jax.ops.index[self.mapping_seed], 1.1
        )

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "max_steps_in_episode": 100,
            "reward_timestep": jnp.array([1, 3, 10, 30, 100]),
            "optimal_return": 1.1,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        state = {
            "rewards": state["rewards"],
            "context": lax.select(state["time"] == 0, action, state["context"]),
            "time": state["time"] + 1,
        }
        reward = lax.select(
            state["time"] == params["reward_timestep"][state["context"]],
            state["rewards"][state["context"]],
            0.0,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        state["terminal"] = done
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        state = {
            "rewards": self.reward,
            "context": -1,
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state, params), state

    def get_obs(self, state: dict, params: dict) -> Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(1, 2), dtype=jnp.float32)
        obs = jax.ops.index_update(obs, jax.ops.index[0, 0], state["context"])
        obs = jax.ops.index_update(
            obs,
            jax.ops.index[0, 1],
            state["time"] / params["max_steps_in_episode"],
        )
        return obs

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done = state["time"] == params["max_steps_in_episode"]
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiscountingChain-v0"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(self.n_actions)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(-1, self.n_actions, (1, 2), dtype=jnp.float32)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "rewards": spaces.Box(1, 1.1, (self.n_actions,), dtype=jnp.float32),
                "context": spaces.Box(-1, self.n_actions, (), dtype=jnp.float32),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

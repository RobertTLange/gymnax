import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
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

    def __init__(self, mapping_seed: int = 0):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict(
            {
                "max_steps_in_episode": 100,
                "reward_timestep": jnp.array([1, 3, 10, 30, 100]),
                "n_actions": 5,
                "mapping_seed": mapping_seed,
                "optimal_return": 1.1,
            }
        )
        # Setup reward fct fron mapping seed - random sampling outside of env
        reward = jnp.ones(self.env_params["n_actions"])
        reward = jax.ops.index_update(reward, jax.ops.index[mapping_seed], 1.1)
        self.update_env_params("rewards", reward)

    def step(
        self, key: PRNGKey, state: dict, action: int
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        state = {
            "rewards": state["rewards"],
            "context": lax.select(state["time"] == 0, action, state["context"]),
            "time": state["time"] + 1,
        }
        reward = lax.select(
            state["time"] == self.env_params["reward_timestep"][state["context"]],
            state["rewards"][state["context"]],
            0.0,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state)
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        state = {
            "rewards": self.env_params["rewards"],
            "context": -1,
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(1, 2), dtype=jnp.float32)
        obs = jax.ops.index_update(obs, jax.ops.index[0, 0], state["context"])
        obs = jax.ops.index_update(
            obs,
            jax.ops.index[0, 1],
            state["time"] / self.env_params["max_steps_in_episode"],
        )
        return obs

    def is_terminal(self, state: dict) -> bool:
        """Check whether state is terminal."""
        done = state["time"] == self.env_params["max_steps_in_episode"]
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiscountingChain-v0"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(self.env_params["n_actions"])

    @property
    def observation_space(self):
        """Observation space of the environment."""
        return spaces.Box(-1, self.env_params["n_actions"], (1, 2), dtype=jnp.float32)

    @property
    def state_space(self):
        """State space of the environment."""
        return spaces.Dict(
            {
                "rewards": spaces.Box(
                    1, 1.1, (self.env_params["n_actions"],), dtype=jnp.float32
                ),
                "context": spaces.Box(
                    -1, self.env_params["n_actions"], (), dtype=jnp.float32
                ),
                "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

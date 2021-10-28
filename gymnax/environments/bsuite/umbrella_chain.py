import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class UmbrellaChain(environment.Environment):
    """
    JAX Compatible version of UmbrellaChain bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/umbrella_chain.py
    """

    def __init__(self, n_distractor: int = 0):
        super().__init__()
        self.n_distractor = n_distractor

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "chain_length": 10,
            "max_steps_in_episode": 100,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        has_umbrella = lax.select(state["time"] + 1 == 1, action, state["has_umbrella"])
        reward = 0
        # Check if chain is full/up
        chain_full = state["time"] + 1 == params["chain_length"]
        has_need = has_umbrella == state["need_umbrella"]
        reward += jnp.logical_and(chain_full, has_need)
        reward -= jnp.logical_and(chain_full, 1 - has_need)
        total_regret = state["total_regret"] + 2 * jnp.logical_and(
            chain_full, 1 - has_need
        )

        # If chain is not full/up add random rewards
        key_reward, key_distractor = jax.random.split(key)
        random_rew = 2 * jax.random.bernoulli(key_reward, p=0.5, shape=()) - 1
        reward += (1 - chain_full) * random_rew

        state = {
            "need_umbrella": jnp.int32(state["need_umbrella"]),
            "has_umbrella": jnp.int32(has_umbrella),
            "total_regret": jnp.int32(total_regret),
            "time": state["time"] + 1,
        }
        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        state["terminal"] = done
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state, key_distractor, params)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        key_need, key_has, key_distractor = jax.random.split(key, 3)
        need_umbrella = jnp.int32(jax.random.bernoulli(key_need, p=0.5, shape=()))
        has_umbrella = jnp.int32(jax.random.bernoulli(key_has, p=0.5, shape=()))
        state = {
            "need_umbrella": need_umbrella,
            "has_umbrella": has_umbrella,
            "total_regret": 0,
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state, key_distractor, params), state

    def get_obs(self, state: dict, key: PRNGKey, params: dict) -> Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(1, 3 + self.n_distractor), dtype=jnp.float32)
        obs = jax.ops.index_update(obs, jax.ops.index[0, 0], state["need_umbrella"])
        obs = jax.ops.index_update(obs, jax.ops.index[0, 1], state["has_umbrella"])
        obs = jax.ops.index_update(
            obs,
            jax.ops.index[0, 2],
            1 - state["time"] / params["chain_length"],
        )
        obs = jax.ops.index_update(
            obs,
            jax.ops.index[0, 3:],
            jax.random.bernoulli(key, p=0.5, shape=(self.n_distractor,)),
        )
        return obs

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done_steps = state["time"] > params["max_steps_in_episode"]
        done_chain = state["time"] == params["chain_length"]
        return jnp.logical_or(done_steps, done_chain)

    @property
    def name(self) -> str:
        """Environment name."""
        return "UmbrellaChain-bsuite"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(0, 1, (1, 3 + self.n_distractor))

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "need_umbrella": spaces.Discrete(2),
                "has_umbrella": spaces.Discrete(2),
                "total_regret": spaces.Discrete(1000),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

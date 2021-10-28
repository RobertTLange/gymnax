import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class MemoryChain(environment.Environment):
    """
    JAX Compatible version of MemoryChain bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/memory_chain.py
    """

    def __init__(self, num_bits: int = 1):
        super().__init__()
        self.num_bits = num_bits

    @property
    def default_params(self):
        # Default environment parameters
        return {"memory_length": 5, "max_steps_in_episode": 1000}

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        obs = self.get_obs(state, params)

        # State smaller than mem length = 0 reward
        reward = 0
        mem_not_full = state["time"] < params["memory_length"]
        correct_action = action == state["context"][state["query"]]
        mem_correct = jnp.logical_and(1 - mem_not_full, correct_action)
        mem_wrong = jnp.logical_and(1 - mem_not_full, 1 - correct_action)
        reward = reward + mem_correct - mem_wrong

        # Update episode loggers
        state = {
            "context": jnp.int32(state["context"]),
            "query": jnp.int32(state["query"]),
            "total_perfect": state["total_perfect"] + mem_correct,
            "total_regret": jnp.float32(state["total_regret"] + 2 * mem_wrong),
            "time": state["time"] + 1,
        }

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        state["terminal"] = done
        info = {"discount": self.discount(state, params)}
        return (lax.stop_gradient(obs), lax.stop_gradient(state), reward, done, info)

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        key_context, key_query = jax.random.split(key)
        context = jax.random.bernoulli(key_context, p=0.5, shape=(self.num_bits,))
        query = jax.random.randint(key_query, minval=0, maxval=self.num_bits, shape=())
        state = {
            "context": jnp.int32(context),
            "query": jnp.int32(query),
            "total_perfect": 0,
            "total_regret": jnp.float32(0),
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state, params), state

    def get_obs(self, state: dict, params: dict) -> Array:
        """Return observation from raw state trafo."""
        # Obs: [time remaining, query, num_bits of context]
        obs = jnp.zeros(shape=(1, self.num_bits + 2), dtype=jnp.float32)
        # Show time remaining - every step.
        obs = jax.ops.index_update(
            obs,
            jax.ops.index[0, 0],
            1 - state["time"] / params["memory_length"],
        )
        # Show query - only last step.
        query_val = lax.select(
            state["time"] == params["memory_length"] - 1, state["query"], 0
        )
        obs = jax.ops.index_update(obs, jax.ops.index[0, 1], query_val)
        # Show context - only first step.
        context_val = lax.select(
            state["time"] == 0, (2 * state["context"] - 1).squeeze(), 0
        )
        obs = jax.ops.index_update(obs, jax.ops.index[0, 2:], context_val)
        return obs

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done_steps = state["time"] > params["max_steps_in_episode"]
        done_mem = state["time"] - 1 == params["memory_length"]
        return jnp.logical_or(done_steps, done_mem)

    @property
    def name(self) -> str:
        """Environment name."""
        return "MemoryChain-bsuite"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(
            0,
            2 * self.num_bits,
            (1, self.num_bits + 2),
            jnp.float32,
        )

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "context": spaces.Discrete(2),
                "query": spaces.Discrete(self.num_bits),
                "total_perfect": spaces.Discrete(params["max_steps_in_episode"]),
                "total_regret": spaces.Discrete(params["max_steps_in_episode"]),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

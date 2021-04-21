import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey


class MemoryChain(environment.Environment):
    def __init__(self, memory_length: int=5,
                 num_bits: int=1):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict(
                            {"memory_length": 5,
                             "num_bits": 1,
                             "max_steps_in_episode": 1000})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        obs = self.get_obs(state)
        state["time"] += 1

        # State smaller than mem length = 0 reward
        reward = 0
        mem_not_full = (state["time"] - 1 < self.env_params["memory_length"])
        correct_action = (action == state["context"][state["query"]])
        mem_correct = jnp.logical_and(1 - mem_not_full, correct_action)
        mem_wrong = jnp.logical_and(1 - mem_not_full, 1-correct_action)
        reward = reward + mem_correct - mem_wrong

        # Update episode loggers
        state["total_perfect"] +=  mem_correct
        state["total_regret"] += 2*mem_wrong

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state)
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (lax.stop_gradient(obs),
                lax.stop_gradient(state), reward, done, info)

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        key_context, key_query = jax.random.split(key)
        context = jax.random.bernoulli(key_context, p=0.5,
                                       shape=(self.env_params["num_bits"],))
        query = jax.random.randint(key_query, minval=0,
                                   maxval=self.env_params["num_bits"],
                                   shape=())
        state = {"context": context,
                 "query": query,
                 "total_perfect": 0,
                 "total_regret": 0,
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        # Obs: [time remaining, query, num_bits of context]
        obs = jnp.zeros(shape=(1, self.env_params["num_bits"] + 2),
                        dtype=jnp.float32)
        # Show time remaining - every step.
        obs = jax.ops.index_update(obs, jax.ops.index[0, 0],
                                   1 - state["time"] / self.env_params["memory_length"])
        # Show query - only last step.
        query_val = lax.select(state["time"] ==
                               self.env_params["memory_length"] - 1,
                               state["query"], 0)
        obs = jax.ops.index_update(obs, jax.ops.index[0, 1], query_val)
        # Show context - only first step.
        context_val = lax.select(state["time"] == 0,
                                 (2*state["context"]-1).squeeze(), 0)
        obs = jax.ops.index_update(obs, jax.ops.index[0, 2:], context_val)
        return obs

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        done_mem = (state["time"] - 1 < self.env_params["memory_length"])
        return jnp.logical_and(done_steps, done_mem)

    @property
    def name(self) -> str:
        """ Environment name. """
        return "MemoryChain-bsuite"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(2)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(0, 2*self.env_params["num_bits"],
                          (1, self.env_params["num_bits"]+2),
                          jnp.float32)

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"context": spaces.Discrete(2),
             "query": spaces.Discrete(self.env_params["num_bits"]),
             "total_perfect": spaces.Discrete(
                                    self.env_params["max_steps_in_episode"]),
             "total_regret": spaces.Discrete(
                                    self.env_params["max_steps_in_episode"]),
             "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})

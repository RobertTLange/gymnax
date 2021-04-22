import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey


class UmbrellaChain(environment.Environment):
    """
    JAX Compatible version of UmbrellaChain bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/umbrella_chain.py
    """
    def __init__(self, chain_length: int = 10,
                 n_distractor: int = 0):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"chain_length": chain_length,
                                      "n_distractor": n_distractor,
                                      "max_steps_in_episode": 100})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        state["time"] += 1
        state["has_umbrella"] = lax.select(state["time"] == 1, action,
                                           state["has_umbrella"])
        reward = 0
        # Check if chain is full/up
        chain_full = (state["time"] == self.env_params["chain_length"])
        has_need = (state["has_umbrella"] == state["need_umbrella"])
        reward += jnp.logical_and(chain_full, has_need)
        reward -= jnp.logical_and(chain_full, 1-has_need)
        state["total_regret"] += 2* jnp.logical_and(chain_full, 1-has_need)

        # If chain is not full/up add random rewards
        key_reward, key_distractor = jax.random.split(key)
        random_rew = 2*jax.random.bernoulli(key_reward, p=0.5, shape=()) - 1
        reward += (1 - chain_full) * random_rew
        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state)
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (lax.stop_gradient(self.get_obs(state, key_distractor)),
                lax.stop_gradient(state), reward, done, info)

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        key_need, key_has, key_distractor = jax.random.split(key, 3)
        need_umbrella = jax.random.bernoulli(key_need, p=0.5, shape=())
        has_umbrella = jax.random.bernoulli(key_has, p=0.5, shape=())
        state = {"need_umbrella": need_umbrella,
                 "has_umbrella": has_umbrella,
                 "total_regret": 0,
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state, key_distractor), state

    def get_obs(self, state: dict, key: PRNGKey) -> Array:
        """ Return observation from raw state trafo. """
        obs = jnp.zeros(shape=(1, 3 + self.env_params["n_distractor"]),
                        dtype=jnp.float32)
        obs = jax.ops.index_update(obs, jax.ops.index[0, 0],
                                   state["need_umbrella"])
        obs = jax.ops.index_update(obs, jax.ops.index[0, 1],
                                   state["has_umbrella"])
        obs = jax.ops.index_update(obs, jax.ops.index[0, 2],
                                   1 - state["time"] /
                                   self.env_params["chain_length"])
        obs = jax.ops.index_update(obs, jax.ops.index[0, 3:],
                                   jax.random.bernoulli(key, p=0.5,
                                   shape=(self.env_params["n_distractor"],)))
        return obs

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        done_chain = (state["time"] == self.env_params["chain_length"])
        return jnp.logical_or(done_steps, done_chain)

    @property
    def name(self) -> str:
        """ Environment name. """
        return "UmbrellaChain-bsuite"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(2)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(0, 1, (1, 3 + self.env_params["n_distractor"]))

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"need_umbrella": spaces.Discrete(2),
             "has_umbrella": spaces.Discrete(2),
             "total_regret": spaces.Discrete(1000),
             "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})

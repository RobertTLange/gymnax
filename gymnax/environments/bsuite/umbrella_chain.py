"""JAX Compatible version of UmbrellaChain bsuite environment.


Source:
github.com/deepmind/bsuite/blob/master/bsuite/environments/umbrella_chain.py
"""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    need_umbrella: jnp.int32
    has_umbrella: jnp.int32
    total_regret: jnp.int32
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    chain_length: int = 10
    max_steps_in_episode: int = 100


class UmbrellaChain(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of UmbrellaChain bsuite environment."""

    def __init__(self, n_distractor: int = 0):
        super().__init__()
        self.n_distractor = n_distractor

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        has_umbrella = lax.select(state.time + 1 == 1, action, state.has_umbrella)
        reward = 0
        # Check if chain is full/up
        chain_full = state.time + 1 == params.chain_length
        has_need = has_umbrella == state.need_umbrella
        reward += jnp.logical_and(chain_full, has_need)
        reward -= jnp.logical_and(chain_full, 1 - has_need)
        total_regret = state.total_regret + 2 * jnp.logical_and(
            chain_full, 1 - has_need
        )

        # If chain is not full/up add random rewards
        key_reward, key_distractor = jax.random.split(key)
        random_rew = 2 * jax.random.bernoulli(key_reward, p=0.5, shape=()) - 1
        reward += (1 - chain_full) * random_rew

        state = EnvState(
            need_umbrella=jnp.int32(state.need_umbrella),
            has_umbrella=jnp.int32(has_umbrella),
            total_regret=jnp.int32(total_regret),
            time=state.time + 1,
        )
        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(
                self.get_obs(state=state, key=key_distractor, params=params)
            ),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        key_need, key_has, key_distractor = jax.random.split(key, 3)
        need_umbrella = jnp.int32(jax.random.bernoulli(key_need, p=0.5, shape=()))
        has_umbrella = jnp.int32(jax.random.bernoulli(key_has, p=0.5, shape=()))
        state = EnvState(
            need_umbrella=need_umbrella,
            has_umbrella=has_umbrella,
            total_regret=0,
            time=0,
        )
        return self.get_obs(state=state, key=key_distractor, params=params), state

    def get_obs(
        self, state: EnvState, key: chex.PRNGKey, params: EnvParams
    ) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(3 + self.n_distractor,), dtype=jnp.float32)
        obs = obs.at[0].set(state.need_umbrella)
        obs = obs.at[1].set(state.has_umbrella)
        obs = obs.at[2].set(1 - state.time / params.chain_length)
        obs = obs.at[3:].set(
            jax.random.bernoulli(key, p=0.5, shape=(self.n_distractor,)),
        )
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        done_chain = state.time == params.chain_length
        return jnp.logical_or(done_steps, done_chain)

    @property
    def name(self) -> str:
        """Environment name."""
        return "UmbrellaChain-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (3 + self.n_distractor,))

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "need_umbrella": spaces.Discrete(2),
                "has_umbrella": spaces.Discrete(2),
                "total_regret": spaces.Discrete(1000),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

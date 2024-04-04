"""JAX compatible version of DiscountingChain bsuite environment.


Source:
github.com/deepmind/bsuite/blob/master/bsuite/environments/discounting_chain.py.
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple, Union
import chex
from flax import struct
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    rewards: chex.Array
    context: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    reward_timestep: chex.Array = dataclasses.field(
        default_factory=lambda: jnp.array([1, 3, 10, 30, 100])
    )
    optimal_return: float = 1.1
    max_steps_in_episode: int = 100


class DiscountingChain(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of DiscountingChain bsuite environment."""

    def __init__(self, n_actions: int = 5, mapping_seed: int = 0):
        super().__init__()
        self.n_actions = n_actions
        self.mapping_seed = mapping_seed

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(reward_timestep=jnp.array([1, 3, 10, 30, 100]))

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        state = EnvState(
            rewards=state.rewards,
            context=lax.select(state.time == 0, action, state.context),
            time=state.time + 1,
        )
        reward = lax.select(
            state.time == params.reward_timestep[state.context],
            state.rewards[state.context],
            0.0,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Setup reward fct from mapping seed - random sampling outside of env
        reward = (
            jnp.ones(self.n_actions).at[self.mapping_seed].set(params.optimal_return)
        )
        state = EnvState(rewards=reward, context=jnp.array(-1), time=0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(2,), dtype=jnp.float32)
        obs = obs.at[0].set(state.context)
        obs = obs.at[1].set(
            state.time / params.max_steps_in_episode,
        )
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiscountingChain-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.n_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-1, self.n_actions, (2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "rewards": spaces.Box(
                    1,
                    params.optimal_return,
                    (self.n_actions,),
                    dtype=jnp.float32,
                ),
                "context": spaces.Box(-1, self.n_actions, (), dtype=jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

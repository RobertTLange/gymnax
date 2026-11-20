"""JAX implementation of the bsuite DiscountingChain environment."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the DiscountingChain environment.

    Attributes:
        rewards: Reward paid by each chain, shape `(n_actions,)`.
        context: Index of the chain selected by the first action of the episode, -1
            before the first step.
        time: Number of steps taken in the current episode.
    """

    rewards: jax.Array
    context: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the DiscountingChain environment.

    Attributes:
        reward_timestep: Step at which each chain pays its reward, one entry per
            action.
        optimal_return: Reward paid by the chain with index `mapping_seed`. All other
            chains pay 1.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    reward_timestep: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.array([1, 3, 10, 30, 100])
    )
    optimal_return: float = 1.1
    max_steps_in_episode: int = 100


class DiscountingChain(environment.Environment[EnvState, EnvParams]):
    """Choose between chains that pay their reward after different delays.

    JAX port of the bsuite DiscountingChain environment. The first action of an
    episode selects one of `n_actions` chains and all later actions are ignored.
    Each chain pays a single reward after its own delay, which tests how an agent
    values rewards that arrive at different times. Resets are deterministic.

    - Observation: `Box` of shape `(2,)` with the index of the selected chain (-1
      before the first step) and the fraction `time / params.max_steps_in_episode`
      of the episode that has elapsed.
    - Actions: `Discrete(n_actions)`. The first action selects the chain.
    - Reward: the reward of the selected chain at step
      `params.reward_timestep[chain]`, 0 on all other steps. The chain with index
      `mapping_seed` pays `params.optimal_return` (1.1 by default), all others 1.
    - Termination: never.
    - Truncation: after `params.max_steps_in_episode` steps (100 by default).

    Args:
        n_actions: Number of actions and chains. `params.reward_timestep` needs one
            entry per action; the default has 5.
        mapping_seed: Index of the chain that pays `params.optimal_return`.

    References:
        - [Osband et al. (2020)](https://arxiv.org/abs/1908.03568)
        - [bsuite implementation](https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/discounting_chain.py)
    """

    def __init__(self, n_actions: int = 5, mapping_seed: int = 0):
        super().__init__()
        self.n_actions = n_actions
        self.mapping_seed = mapping_seed

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for DiscountingChain-bsuite."""
        return EnvParams(reward_timestep=jnp.array([1, 3, 10, 30, 100]))

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        state = EnvState(
            rewards=state.rewards,
            context=jax.lax.select(state.time == 0, action, state.context),
            time=state.time + 1,
        )
        reward = jax.lax.select(
            state.time == params.reward_timestep[state.context],
            state.rewards[state.context],
            0.0,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state deterministically; `key` is not used."""
        # Setup reward fct from mapping seed - random sampling outside of env
        reward = (
            jnp.ones(self.n_actions).at[self.mapping_seed].set(params.optimal_return)
        )
        state = EnvState(rewards=reward, context=jnp.array(-1), time=0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(2,), dtype=jnp.float32)
        obs = obs.at[0].set(state.context)
        obs = obs.at[1].set(
            state.time / params.max_steps_in_episode,
        )
        return obs

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiscountingChain-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
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

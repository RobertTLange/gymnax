"""JAX implementation of the bsuite SimpleBandit environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the SimpleBandit environment.

    Attributes:
        rewards: Reward of each action, shape `(num_actions,)`. A random permutation
            of `num_actions` evenly spaced values from 0 to 1.
        total_regret: Sum of `params.optimal_return - reward` over the steps of the
            current episode.
        time: Number of steps taken in the current episode.
    """

    rewards: float
    total_regret: float
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the SimpleBandit environment.

    Attributes:
        optimal_return: Reward of the best action, used to compute the regret.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    optimal_return: float = 1.0
    max_steps_in_episode: int = 100


class SimpleBandit(environment.Environment[EnvState, EnvParams]):
    """Pull the arm with the highest reward in a deterministic multi-armed bandit.

    JAX port of the bsuite bandit environment. Each reset assigns `num_actions`
    evenly spaced rewards from 0 to 1 to the actions in a random order. Every step
    is a complete episode, so `step` redraws the assignment after each action.

    - Observation: `Box` of shape `(1, 1)` that always contains 1.
    - Actions: `Discrete(num_actions)`. The index of the arm to pull.
    - Reward: the reward assigned to the chosen action, a value from 0 to 1.
    - Termination: after every step.
    - Truncation: after `params.max_steps_in_episode` steps (100 by default). Not
      reached with the default, since every episode terminates after one step.

    Args:
        num_actions: Number of arms.

    References:
        - [Osband et al. (2020)](https://arxiv.org/abs/1908.03568)
        - [bsuite implementation](https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/bandit.py)
    """

    def __init__(self, num_actions: int = 11):
        super().__init__()
        self.n_actions = num_actions

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for SimpleBandit-bsuite."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        reward = state.rewards[action]
        state = EnvState(
            rewards=state.rewards,
            total_regret=state.total_regret + params.optimal_return - reward,
            time=state.time + 1,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, Any]:  # dict]:
        """Reset environment state by sampling a new reward for each action."""
        action_mask = jax.random.choice(
            key,
            jnp.arange(self.num_actions),
            shape=(self.num_actions,),
            replace=False,
        )
        rewards = jnp.linspace(0, 1, self.num_actions)[action_mask]

        state = EnvState(rewards=rewards, total_regret=0.0, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return the constant observation, which does not depend on the state."""
        return jnp.ones(shape=(1, 1), dtype=jnp.float32)

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Episode always terminates after single step - Do not reset though!
        return jnp.array(True)

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleBandit-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(1, 1, (1, 1))

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "rewards": spaces.Box(0, 1, (self.num_actions,)),
                "total_regret": spaces.Box(0, params.max_steps_in_episode, ()),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

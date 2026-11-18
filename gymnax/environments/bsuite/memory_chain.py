"""JAX implementation of the bsuite MemoryChain environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the MemoryChain environment.

    Attributes:
        context: Context bits to remember, an array of `num_bits` values in
            `{0, 1}`.
        query: Index of the context bit the agent is asked for.
        total_perfect: Number of correct answers in the current episode.
        total_regret: Regret of the current episode. A wrong answer adds 2.
        time: Number of steps taken in the current episode.
    """

    context: int
    query: int
    total_perfect: int
    total_regret: float
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the MemoryChain environment.

    Attributes:
        memory_length: Number of steps before the step in which the agent answers
            the query.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    memory_length: int = 5
    max_steps_in_episode: int = 1000


class MemoryChain(environment.Environment[EnvState, EnvParams]):
    """Remember a context bit and report it at the end of the episode.

    JAX port of the bsuite MemoryChain environment, a test of memory. Each reset
    samples `num_bits` context bits and the index of the bit that is queried, both
    uniformly at random. The context is only observed at the start of the episode
    and the query only before the last step, in which the action is the answer.

    `step_env` returns the observation of the state before the transition, so the
    first step repeats the observation returned by the reset.

    - Observation: `Box` of shape `(num_bits + 2,)`. Entry 0 is the remaining time
      `1 - time / params.memory_length`. Entry 1 is the query index when
      `time == params.memory_length - 1` and 0 otherwise. The other entries are the
      context bits encoded as -1 and 1 when `time == 0` and 0 otherwise.
    - Actions: `Discrete(2)`. The value of the queried context bit. Actions before
      the last step are ignored.
    - Reward: on the last step, 1 if the action equals the queried context bit and
      -1 otherwise. 0 on all other steps.
    - Termination: after `params.memory_length + 1` steps (6 by default).
    - Truncation: after `params.max_steps_in_episode` steps (1000 by default).

    Args:
        num_bits: Number of context bits.

    References:
        - [Osband et al. (2020)](https://arxiv.org/abs/1908.03568)
        - [bsuite implementation](https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/memory_chain.py)
    """

    def __init__(self, num_bits: int = 1):
        super().__init__()
        self.num_bits = num_bits

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for MemoryChain-bsuite."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        obs = self.get_obs(state, params)

        # State smaller than mem length = 0 reward
        reward = 0.0
        mem_not_full = state.time < params.memory_length
        correct_action = action == state.context[state.query]
        mem_correct = jnp.logical_and(1 - mem_not_full, correct_action)
        mem_wrong = jnp.logical_and(1 - mem_not_full, 1 - correct_action)
        reward = reward + mem_correct - mem_wrong

        # Update episode loggers
        state = EnvState(
            context=jnp.int32(state.context),
            query=jnp.int32(state.query),
            total_perfect=jnp.int32(state.total_perfect + mem_correct),
            total_regret=jnp.float32(state.total_regret + 2 * mem_wrong),
            time=state.time + 1,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling a new context and query."""
        key_context, key_query = jax.random.split(key)
        context = jax.random.bernoulli(key_context, p=0.5, shape=(self.num_bits,))
        query = jax.random.randint(key_query, minval=0, maxval=self.num_bits, shape=())
        state = EnvState(
            context=jnp.int32(context),
            query=jnp.int32(query),
            total_perfect=jnp.int32(0),
            total_regret=jnp.float32(0),
            time=jnp.int32(0),
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        # Obs: [time remaining, query, num_bits of context]
        obs = jnp.zeros(shape=(self.num_bits + 2,), dtype=jnp.float32)
        # Show time remaining - every step.
        obs = obs.at[0].set(
            1 - state.time / params.memory_length,
        )
        # Show query - only last step.
        query_val = jax.lax.select(
            state.time == params.memory_length - 1,
            state.query,
            jnp.zeros_like(state.query),
        )
        obs = obs.at[1].set(query_val)
        # Show context - only first step.
        context_val = jax.lax.select(
            state.time == 0, 2 * state.context - 1, jnp.zeros_like(state.context)
        )
        obs = obs.at[2:].set(context_val)
        return obs

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        done_mem = state.time - 1 == params.memory_length
        return done_mem

    @property
    def name(self) -> str:
        """Environment name."""
        return "MemoryChain-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            0,
            2 * self.num_bits,
            (self.num_bits + 2,),
            jnp.float32,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "context": spaces.Discrete(2),
                "query": spaces.Discrete(self.num_bits),
                "total_perfect": spaces.Discrete(params.max_steps_in_episode),
                "total_regret": spaces.Discrete(params.max_steps_in_episode),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

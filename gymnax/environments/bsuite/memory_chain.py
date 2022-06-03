import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple
import chex
from flax import struct


@struct.dataclass
class EnvState:
    context: jnp.int32
    query: jnp.int32
    total_perfect: int
    total_regret: jnp.float32
    time: int


@struct.dataclass
class EnvParams:
    memory_length: int = 5
    max_steps_in_episode: int = 1000


class MemoryChain(environment.Environment):
    """
    JAX Compatible version of MemoryChain bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/memory_chain.py
    """

    def __init__(self, num_bits: int = 1):
        super().__init__()
        self.num_bits = num_bits

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        obs = self.get_obs(state, params)

        # State smaller than mem length = 0 reward
        reward = 0
        mem_not_full = state.time < params.memory_length
        correct_action = action == state.context[state.query]
        mem_correct = jnp.logical_and(1 - mem_not_full, correct_action)
        mem_wrong = jnp.logical_and(1 - mem_not_full, 1 - correct_action)
        reward = reward + mem_correct - mem_wrong

        # Update episode loggers
        state = EnvState(
            jnp.int32(state.context),
            jnp.int32(state.query),
            state.total_perfect + mem_correct,
            jnp.float32(state.total_regret + 2 * mem_wrong),
            state.time + 1,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        key_context, key_query = jax.random.split(key)
        context = jax.random.bernoulli(
            key_context, p=0.5, shape=(self.num_bits,)
        )
        query = jax.random.randint(
            key_query, minval=0, maxval=self.num_bits, shape=()
        )
        state = EnvState(
            jnp.int32(context), jnp.int32(query), 0, jnp.float32(0), 0
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # Obs: [time remaining, query, num_bits of context]
        obs = jnp.zeros(shape=(1, self.num_bits + 2), dtype=jnp.float32)
        # Show time remaining - every step.
        obs = obs.at[0, 0].set(
            1 - state.time / params.memory_length,
        )
        # Show query - only last step.
        query_val = lax.select(
            state.time == params.memory_length - 1, state.query, 0
        )
        obs = obs.at[0, 1].set(query_val)
        # Show context - only first step.
        context_val = lax.select(
            state.time == 0, (2 * state.context - 1).squeeze(), 0
        )
        obs = obs.at[0, 2:].set(context_val)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time > params.max_steps_in_episode
        done_mem = state.time - 1 == params.memory_length
        return jnp.logical_or(done_steps, done_mem)

    @property
    def name(self) -> str:
        """Environment name."""
        return "MemoryChain-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            0,
            2 * self.num_bits,
            (1, self.num_bits + 2),
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

"""JAX implementation of DeepSea bsuite environment."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    row: int
    column: int
    bad_episode: bool
    total_bad_episodes: int
    denoised_return: int
    optimal_return: float
    action_mapping: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    deterministic: bool = True
    sample_action_map: bool = False
    unscaled_move_cost: float = 0.01
    randomize_actions: bool = True
    max_steps_in_episode: int = 2000


class DeepSea(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of DeepSea bsuite environment.


    Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py.
    """

    def __init__(self, size: int = 8, action_mapping_key: jax.Array | None = None):
        """Create a DeepSea instance with a fixed, reproducible action mapping."""
        super().__init__()
        self.size = size
        if action_mapping_key is None:
            action_mapping_key = jax.random.key(0)
        self.action_mapping = jax.random.bernoulli(
            action_mapping_key, 0.5, (size, size)
        ).astype(jnp.int32)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.size)

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Step while preserving bsuite's cumulative episode metrics on reset."""
        if params is None:
            params = self.default_params
        key_step, key_reset = jax.random.split(key)
        obs_step, state_step, reward, terminated, info = self.step_env(
            key_step, state, action, params
        )
        truncated = self.is_truncated(state_step, params)
        done = jnp.logical_or(terminated, truncated)
        obs_reset, state_reset = self.reset_env(key_reset, params)
        next_state = jax.tree.map(
            lambda reset, stepped: jax.lax.select(done, reset, stepped),
            state_reset,
            state_step,
        )
        next_state = next_state.replace(
            total_bad_episodes=jax.lax.select(
                done, state_step.total_bad_episodes, next_state.total_bad_episodes
            ),
            denoised_return=jax.lax.select(
                done, state_step.denoised_return, next_state.denoised_return
            ),
        )
        info = {
            **info,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": obs_step,
        }
        obs = jax.tree.map(
            lambda reset_leaf, step_leaf: jax.lax.select(done, reset_leaf, step_leaf),
            obs_reset,
            obs_step,
        )
        return obs, next_state, reward, terminated, truncated, info

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Pull out randomness for easier testing
        key_reward, key_trans = jax.random.split(key)
        rand_reward = jax.random.normal(key_reward, shape=())
        rand_trans_cond = (
            jax.random.uniform(key_trans, shape=(), minval=0, maxval=1) > 1 / self.size
        )

        action_right = action == state.action_mapping[state.row, state.column]
        right_rand_cond = jnp.logical_or(rand_trans_cond, params.deterministic)
        right_cond = jnp.logical_and(action_right, right_rand_cond)

        reward, denoised_return = step_reward(
            state, action_right, right_cond, rand_reward, self.size, params
        )
        column, row, bad_episode = step_transition(
            state, action_right, right_cond, self.size
        )
        state = state.replace(
            row=row,
            column=column,
            bad_episode=bad_episode,
            denoised_return=denoised_return,
            time=state.time + 1,
        )

        # Check row condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        state = state.replace(
            total_bad_episodes=state.total_bad_episodes + done * state.bad_episode
        )
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
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        optimal_no_cost = (1 - params.deterministic) * (1 - 1 / self.size) ** (
            self.size - 1
        ) + params.deterministic * 1.0
        optimal_return = optimal_no_cost - params.unscaled_move_cost

        sampled_action_mapping = jax.random.bernoulli(
            key, 0.5, (self.size, self.size)
        ).astype(jnp.int32)
        debug_action_mapping = jnp.ones((self.size, self.size), dtype=jnp.int32)
        fixed_action_mapping = jnp.where(
            params.randomize_actions,
            self.action_mapping,
            debug_action_mapping,
        )
        action_mapping = jnp.where(
            params.sample_action_map,
            sampled_action_mapping,
            fixed_action_mapping,
        )

        state = EnvState(
            row=0,
            column=0,
            bad_episode=False,
            total_bad_episodes=0,
            denoised_return=0,
            optimal_return=optimal_return,
            action_mapping=action_mapping,
            time=0,
        )

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        obs_end = jnp.zeros(shape=(self.size, self.size), dtype=jnp.float32)
        end_cond = state.row >= self.size
        obs_upd = obs_end.at[state.row, state.column].set(1.0)
        return jax.lax.select(end_cond, obs_end, obs_upd)

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return state.row == self.size

    @property
    def name(self) -> str:
        """Environment name."""
        return "DeepSea-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.size, self.size), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "row": spaces.Discrete(self.size),
                "column": spaces.Discrete(self.size),
                "bad_episode": spaces.Discrete(2),
                "total_bad_episodes": spaces.Discrete(2000),
                "denoised_return": spaces.Box(0, 1000, ()),
                "optimal_return": spaces.Box(0, 1000, ()),
                "action_mapping": spaces.Box(
                    0,
                    1,
                    (self.size, self.size),
                    dtype=jnp.int32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def step_reward(
    state: EnvState,
    action_right: bool,
    right_cond: jax.Array,
    rand_reward: jax.Array,
    size: int,
    params: EnvParams,
) -> tuple[jax.Array, jax.Array]:
    """Get the reward for the selected action."""
    reward = 0.0
    # Reward calculation.
    rew_cond = jnp.logical_and(state.column == size - 1, action_right)
    reward += rew_cond
    denoised_return = state.denoised_return + rew_cond

    # Noisy rewards on the 'end' of chain.
    col_at_edge = jnp.logical_or(state.column == 0, state.column == size - 1)
    chain_end = jnp.logical_and(state.row == size - 1, col_at_edge)
    noisy_chain_end = jnp.logical_and(chain_end, jnp.logical_not(params.deterministic))
    reward += rand_reward * noisy_chain_end
    reward -= action_right * params.unscaled_move_cost / size
    return reward, denoised_return


def step_transition(
    state: EnvState, action_right: bool, right_cond: jax.Array, size: int
) -> tuple[jax.Array, int, jax.Array]:
    """Get the state transition for the selected action."""
    # Standard right path transition
    column = jax.lax.select(
        right_cond, jnp.clip(state.column + 1, 0, size - 1), state.column
    )

    # You were on the right path and went wrong
    right_wrong_cond = jnp.logical_and(1 - action_right, state.row == column)
    bad_episode = jax.lax.select(right_wrong_cond, True, state.bad_episode)
    column = jax.lax.select(
        action_right, column, jnp.clip(state.column - 1, 0, size - 1)
    )
    row = state.row + 1
    return column, row, bad_episode

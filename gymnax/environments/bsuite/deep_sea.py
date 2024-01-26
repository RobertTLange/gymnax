import jax
import jax.numpy as jnp
import chex

from jax import lax
from gymnax.environments import spaces
from typing import Tuple, Optional, Callable
from flax import struct

from .env import Environment

"""
Code adapted from
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/deep_sea.py
"""


@struct.dataclass
class EnvState:
    row: int
    column: int
    bad_episode: bool
    action_mapping: chex.Array
    time: int


last_goal_col_dist = jax.tree_util.Partial(lambda key, size: size - 1)
uniform_action_mapping = jax.tree_util.Partial(
    lambda key, size: jax.random.randint(key, shape=(1,), minval=0, maxval=2)
    * jnp.ones((size, size))
)
default_action_mapping = jax.tree_util.Partial(lambda key, size: jnp.ones((size, size)))


@struct.dataclass
class EnvParams:
    action_mapping: chex.Array
    goal_column: int = None
    unscaled_move_cost: float = 0.01  # the cost of following the optimal path
    max_steps_in_episode: int = 2000


class DeepSea(Environment):
    """
    JAX Compatible version of DeepSea bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py
    """

    def __init__(
        self,
        size: int,
        action_mapping_dist: Callable[
            [chex.PRNGKey], chex.Array
        ] = default_action_mapping,
        goal_column_dist: Callable[[chex.PRNGKey, int], int] = last_goal_col_dist,
    ):
        super().__init__()
        self.size = size
        self.goal_column_dist = goal_column_dist
        self.action_mapping_dist = action_mapping_dist

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(jnp.ones((self.size, self.size)))

    def init_env(
        self, key: chex.PRNGKey, params: EnvParams, meta_params: chex.Array = None
    ) -> EnvParams:
        """Initialize environment state."""
        goal_column = self.goal_column_dist(key, self.size)
        action_mapping = self.action_mapping_dist(key, self.size)
        return params.replace(goal_column=goal_column, action_mapping=action_mapping)

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        action_right = action == state.action_mapping[state.row, state.column]

        reward = step_reward(state, self.size, params)
        column, row, bad_episode = step_transition(
            state, action_right, self.size, params.goal_column
        )
        state = state.replace(
            row=row,
            column=column,
            bad_episode=bad_episode,
            time=state.time + 1,
        )

        # Check row condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def Q_function(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Get Q function."""
        current_reward = step_reward(state, self.size, params)
        _, _, right_bad_episode = step_transition(
            state, True, self.size, params.goal_column
        )
        _, _, left_bad_episode = step_transition(
            state, False, self.size, params.goal_column
        )
        right_q = jax.lax.select(
            right_bad_episode,
            0.0,
            1.0 - params.unscaled_move_cost / self.size * (self.size - 1 - state.row),
        )
        left_q = jax.lax.select(
            left_bad_episode,
            0.0,
            1.0 - params.unscaled_move_cost / self.size * (self.size - 1 - state.row),
        )

        return jnp.array([left_q, right_q]) + current_reward

    def optimal_policy(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ) -> int:
        """Get optimal policy. Choose the action with the highest Q value."""
        # TODO If both actions have the same Q value, choose randomly.
        q_values = self.Q_function(state, params)
        action_ind = jnp.argmax(
            q_values
        )  # jax.random.choice(key, jnp.where(q_values == q_values.max())[0])
        right_action_ind = params.action_mapping[state.row, state.column]
        return jax.lax.stop_gradient(
            (1.0 - jnp.logical_xor(action_ind, right_action_ind)).astype(jnp.int32)
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""

        state = EnvState(0, 0, False, jnp.ones([self.size, self.size]), 0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        obs_end = jnp.zeros(shape=(self.size, self.size), dtype=jnp.float32)
        end_cond = state.row >= self.size
        obs_upd = obs_end.at[state.row, state.column].set(1.0)
        return jax.lax.select(end_cond, obs_end, obs_upd)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_row = state.row == self.size
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_row, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "DeepSea-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
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
                "action_mapping": spaces.Box(
                    0,
                    1,
                    (self.size, self.size),
                    dtype=jnp.int_,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def step_reward(state: EnvState, size: int, params: EnvParams) -> Tuple[float, float]:
    """Get the reward for the selected action."""
    # Reward calculation.
    is_in_optimal_path = in_optimal_path(
        size, state.row, state.column, params.goal_column
    )
    reward = jnp.where(is_in_optimal_path, -params.unscaled_move_cost / size, 0.0)
    reward += jnp.logical_and(state.column == params.goal_column, state.row == size - 1)
    return reward


def step_transition(
    state: EnvState, action_right: bool, size: int, goal_column: int
) -> Tuple[int, int, int]:
    """Get the state transition for the selected action."""
    # Standard right path transition
    column = jax.lax.select(
        action_right,
        jnp.clip(state.column + 1, 0, size - 1),
        jnp.clip(state.column - 1, 0, size - 1),
    )
    row = state.row + 1

    # if it is not possible to reach the goal
    not_reach_cond = jnp.logical_not(in_optimal_path(size, row, column, goal_column))
    bad_episode = jax.lax.select(not_reach_cond, True, state.bad_episode)

    return column, row, bad_episode


def in_optimal_path(size: int, row: int, column: int, goal_column: int) -> bool:
    """Check whether the current position is in the optimal path."""
    return jnp.logical_and(
        goal_column >= column - (size - 1 - row),
        goal_column <= column + (size - 1 - row),
    )

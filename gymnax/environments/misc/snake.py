"""JAX implementation of the Snake (apple-eating) environment."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jax.custom_batching import custom_vmap

from gymnax.environments import environment, spaces

# Actions: 0=up, 1=right, 2=down, 3=left. Vectors are (row, col).
DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the Snake environment.

    Attributes:
        board: Integer array of shape `(height, width)`. A positive entry is the
            number of steps until the snake vacates the cell, so the head cell
            equals `length`. Free cells are 0.
        head: Position of the head as `(row, column)`.
        direction: Last executed action.
        apple: Position of the apple as `(row, column)`.
        length: Number of cells occupied by the snake.
        alive: Whether the snake is alive. `False` after a wall or self
            collision.
        time: Number of steps taken in the current episode.
    """

    board: jax.Array
    head: jax.Array
    direction: jax.Array
    apple: jax.Array
    length: jax.Array
    alive: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the Snake environment.

    Attributes:
        reward_apple: Reward for eating an apple.
        reward_death: Reward for a wall or self collision.
        reward_step: Reward added at every step.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    reward_apple: float = 1.0
    reward_death: float = -1.0
    reward_step: float = 0.0
    max_steps_in_episode: int = 1000


class Snake(environment.Environment[EnvState, EnvParams]):
    """Steer a growing snake to eat apples on a fixed-size grid.

    The snake starts with length 1 at a uniformly random cell with a random
    direction, and the apple at a uniformly random other cell. Eating an apple
    increases the length by one and places a new apple on a uniformly random free
    cell. The current length is returned as `info["length"]`.

    - Observation: `Box` of shape `(height, width, 3)` with binary channels for
      the body, the head and the apple.
    - Actions: `Discrete(4)`. 0 moves up, 1 right, 2 down and 3 left. An action
      opposite to the current direction is ignored while the length exceeds 1.
    - Reward: `params.reward_apple` for eating an apple, plus
      `params.reward_death` for a wall or self collision, plus
      `params.reward_step` at every step.
    - Termination: wall or self collision, or the board is completely filled.
    - Truncation: after `params.max_steps_in_episode` steps (1000 by default).

    Args:
        height: Number of rows of the board. Must be at least 2.
        width: Number of columns of the board. Must be at least 2.
    """

    def __init__(self, height: int = 10, width: int = 10):
        super().__init__()
        if height < 2 or width < 2:
            raise ValueError(
                f"Board dimensions must be at least 2x2, got {height}x{width}"
            )
        self.height = height
        self.width = width
        self.obs_shape = (height, width, 3)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Snake-misc."""
        return EnvParams()

    def _sample_apple(self, key: jax.Array, board: jax.Array) -> jax.Array:
        """Uniformly sample a free cell (uniform over all cells if board is full)."""
        free = (board == 0).reshape(-1).astype(jnp.float32)
        free = jnp.where(free.sum() > 0, free, jnp.ones_like(free))
        idx = jax.random.choice(key, free.shape[0], p=free / free.sum())
        row = idx // self.width
        col = idx % self.width
        return jnp.stack([row, col]).astype(jnp.int32)

    def _update_apple(self, key: jax.Array, state: EnvState) -> jax.Array:
        """Respawn consumed apples, skipping sampling when the whole batch can wait."""

        @custom_vmap
        def update(key, state):
            consumed = jnp.all(state.head == state.apple, axis=-1)
            needs_apple = consumed & (state.length < self.height * self.width)
            sample = self._sample_apple
            for _ in range(state.board.ndim - 2):
                sample = jax.vmap(sample)
            return jax.lax.cond(
                jnp.any(needs_apple),
                lambda: jnp.where(
                    needs_apple[..., None], sample(key, state.board), state.apple
                ),
                lambda: state.apple,
            )

        @update.def_vmap
        def update_batch(axis_size, in_batched, key, state):
            # Reapply the rule so nested vmaps retain one aggregate conditional.
            key, state = jax.tree.map(
                lambda batched, value: (
                    value
                    if batched
                    else jnp.broadcast_to(value, (axis_size, *jnp.shape(value)))
                ),
                tuple(in_batched),
                (key, state),
            )
            return update(key, state), True

        return update(key, state)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        action = jnp.asarray(action, dtype=jnp.int32)
        reverse = jnp.logical_and(
            (action - state.direction) % 4 == 2,
            state.length > 1,
        )
        direction = jnp.where(reverse, state.direction, action)

        bounds = jnp.array([self.height, self.width], dtype=jnp.int32)
        new_head = state.head + DIRECTIONS[direction]
        out_of_bounds = jnp.logical_or(
            jnp.any(new_head < 0), jnp.any(new_head >= bounds)
        )
        new_head = jnp.clip(new_head, 0, bounds - 1)

        ate = jnp.logical_and(
            jnp.logical_not(out_of_bounds), jnp.all(new_head == state.apple)
        )

        # Snake advances: every cell ages one tick unless growing (tail stays).
        board = jnp.where(ate, state.board, jnp.maximum(state.board - 1, 0))
        hit_self = board[new_head[0], new_head[1]] > 0
        dead = jnp.logical_or(out_of_bounds, hit_self)

        length = state.length + ate.astype(jnp.int32)
        board = board.at[new_head[0], new_head[1]].set(length)

        reward = (
            ate * params.reward_apple + dead * params.reward_death + params.reward_step
        ).astype(jnp.float32)

        state = EnvState(
            board=board,
            head=new_head,
            direction=direction,
            apple=state.apple,
            length=length,
            alive=jnp.logical_not(dead),
            time=state.time + 1,
        )
        state = replace(state, apple=self._update_apple(key, state))
        terminated = self.is_terminated(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            terminated,
            {"discount": self.discount(state, params), "length": state.length},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Performs resetting of environment."""
        key_row, key_col, key_dir, key_apple = jax.random.split(key, 4)
        head = jnp.stack(
            [
                jax.random.randint(key_row, (), 0, self.height),
                jax.random.randint(key_col, (), 0, self.width),
            ]
        ).astype(jnp.int32)
        board = jnp.zeros((self.height, self.width), dtype=jnp.int32)
        board = board.at[head[0], head[1]].set(1)
        # Skip the head index in a uniform draw over the remaining cells.
        head_index = head[0] * self.width + head[1]
        apple_index = jax.random.randint(
            key_apple, (), 0, self.height * self.width - 1, dtype=jnp.int32
        )
        apple_index += (apple_index >= head_index).astype(jnp.int32)
        apple = jnp.stack([apple_index // self.width, apple_index % self.width])
        state = EnvState(
            board=board,
            head=head,
            direction=jax.random.randint(key_dir, (), 0, 4, dtype=jnp.int32),
            apple=apple,
            length=jnp.int32(1),
            alive=jnp.bool_(True),
            time=jnp.int32(0),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        head = jnp.zeros((self.height, self.width), dtype=jnp.float32)
        head = head.at[state.head[0], state.head[1]].set(1.0)
        apple = jnp.zeros((self.height, self.width), dtype=jnp.float32)
        apple = apple.at[state.apple[0], state.apple[1]].set(1.0)
        body = (state.board > 0).astype(jnp.float32) - head
        return jnp.stack([body, head, apple], axis=-1)

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is naturally terminated (collision or board filled)."""
        done_dead = jnp.logical_not(state.alive)
        done_full = state.length >= self.height * self.width
        return jnp.logical_or(done_dead, done_full)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Snake-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0.0, 1.0, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        n = self.height * self.width
        return spaces.Dict(
            {
                "board": spaces.Box(0, n, (self.height, self.width), jnp.int32),
                "head": spaces.Box(
                    0,
                    jnp.array([self.height - 1, self.width - 1]),
                    (2,),
                    jnp.int32,
                ),
                "direction": spaces.Discrete(4),
                "apple": spaces.Box(
                    0,
                    jnp.array([self.height - 1, self.width - 1]),
                    (2,),
                    jnp.int32,
                ),
                "length": spaces.Discrete(n + 1),
                "alive": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

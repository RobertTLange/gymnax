"""JAX implementation of the Snake (apple-eating) environment for gymnax."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces

# Actions: 0=up, 1=right, 2=down, 3=left. Vectors are (row, col).
DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


@struct.dataclass
class EnvState(environment.EnvState):
    board: jax.Array  # (H, W) int32; >0 marks a snake cell, value = ticks until it vacates (head = length)
    head: jax.Array  # (2,) int32, (row, col)
    direction: jax.Array  # int32 scalar, last executed action
    apple: jax.Array  # (2,) int32, (row, col)
    length: jax.Array  # int32 scalar
    alive: jax.Array  # bool scalar, False after wall/self collision
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    reward_apple: float = 1.0
    reward_death: float = -1.0
    reward_step: float = 0.0
    max_steps_in_episode: int = 1000


class Snake(environment.Environment[EnvState, EnvParams]):
    """Snake on a fixed-size grid.

    Observation: (H, W, 3) float32 with channels [body, head, apple].
    Actions: 0=up, 1=right, 2=down, 3=left. An action opposite to the current
    direction is ignored while length > 1.
    Termination: wall/self collision, or board completely filled.
    Truncation: ``params.max_steps_in_episode``.
    """

    def __init__(self, height: int = 10, width: int = 10):
        super().__init__()
        self.height = height
        self.width = width
        self.obs_shape = (height, width, 3)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def _sample_apple(self, key: jax.Array, board: jax.Array) -> jax.Array:
        """Uniformly sample a free cell (uniform over all cells if board is full)."""
        free = (board == 0).reshape(-1).astype(jnp.float32)
        free = jnp.where(free.sum() > 0, free, jnp.ones_like(free))
        idx = jax.random.choice(key, free.shape[0], p=free / free.sum())
        return jnp.stack([idx // self.width, idx % self.width]).astype(jnp.int32)

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
        apple = jnp.where(ate, self._sample_apple(key, board), state.apple)

        reward = (
            ate * params.reward_apple
            + dead * params.reward_death
            + params.reward_step
        ).astype(jnp.float32)

        state = EnvState(
            board=board,
            head=new_head,
            direction=direction,
            apple=apple,
            length=length,
            alive=jnp.logical_not(dead),
            time=state.time + 1,
        )
        terminated = self.is_terminal(state, params)

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
        state = EnvState(
            board=board,
            head=head,
            direction=jax.random.randint(key_dir, (), 0, 4, dtype=jnp.int32),
            apple=self._sample_apple(key_apple, board),
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

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Collision, or board completely filled."""
        done_dead = jnp.logical_not(state.alive)
        done_full = state.length >= self.height * self.width
        return jnp.logical_or(done_dead, done_full)

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        return "Snake"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        n = self.height * self.width
        return spaces.Dict(
            {
                "board": spaces.Box(0, n, (self.height, self.width), jnp.int32),
                "head": spaces.Box(0, max(self.height, self.width), (2,), jnp.int32),
                "direction": spaces.Discrete(4),
                "apple": spaces.Box(0, max(self.height, self.width), (2,), jnp.int32),
                "length": spaces.Discrete(n + 1),
                "alive": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

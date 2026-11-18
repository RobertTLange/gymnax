"""JAX implementation of the bsuite Catch environment."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the Catch environment.

    Attributes:
        ball_x: Column of the ball.
        ball_y: Row of the ball. Starts at 0 and increases by one every step.
        paddle_x: Column of the paddle.
        paddle_y: Row of the paddle, always `rows - 1`.
        prev_done: Whether the ball reached the paddle row in the previous step. If
            set, `step_env` starts from a freshly sampled initial state instead of
            moving the ball and the paddle.
        time: Number of steps taken in the current episode.
    """

    ball_x: jax.Array
    ball_y: jax.Array
    paddle_x: int
    paddle_y: int
    prev_done: bool
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the Catch environment.

    Attributes:
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    max_steps_in_episode: int = 1000


class Catch(environment.Environment[EnvState, EnvParams]):
    """Catch a falling ball by moving a paddle along the bottom row of a grid.

    JAX port of the bsuite Catch environment. The ball starts in a uniformly random
    column of row 0 and falls one row per step. The paddle starts in column
    `columns // 2` of the last row.

    - Observation: `Box` of shape `(rows, columns)` that is 0 everywhere except for
      a 1 at the ball position and a 1 at the paddle position.
    - Actions: `Discrete(3)`. 0 moves the paddle one column to the left, 1 keeps it
      in place and 2 moves it one column to the right. The paddle stays within the
      grid.
    - Reward: on the step in which the ball reaches the last row, 1 if the paddle is
      in the ball's column and -1 otherwise. 0 on all other steps.
    - Termination: the ball reaches the last row, which takes `rows - 1` steps.
    - Truncation: after `params.max_steps_in_episode` steps (1000 by default).

    Args:
        rows: Number of rows of the grid.
        columns: Number of columns of the grid.

    References:
        - [Osband et al. (2020)](https://arxiv.org/abs/1908.03568)
        - [bsuite implementation](https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/catch.py)
    """

    def __init__(self, rows: int = 10, columns: int = 5):
        super().__init__()
        self.rows = rows
        self.columns = columns

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Catch-bsuite."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Sample new init state each step & use if there was a reset!
        ball_x, ball_y, paddle_x, paddle_y = sample_init_state(
            key, self.rows, self.columns
        )
        prev_done = state.prev_done

        # Move the paddle + drop the ball.
        dx = action - 1  # [-1, 0, 1] = Left, no-op, right.
        paddle_x = jax.lax.select(
            prev_done,
            paddle_x,
            jnp.clip(state.paddle_x + dx, 0, self.columns - 1),
        )
        ball_y = jax.lax.select(prev_done, ball_y, state.ball_y + 1)
        ball_x = jax.lax.select(prev_done, ball_x, state.ball_x)
        paddle_y = jax.lax.select(prev_done, paddle_y, state.paddle_y)

        # Rewrite reward as boolean multiplication
        prev_done = ball_y == paddle_y
        catched = paddle_x == ball_x
        reward = prev_done * jax.lax.select(catched, 1.0, -1.0)

        state = state.replace(
            ball_x=ball_x,
            ball_y=ball_y,
            paddle_x=paddle_x,
            paddle_y=paddle_y,
            prev_done=prev_done,
            time=state.time + 1,
        )

        # Check number of steps in episode termination condition
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        ball_x, ball_y, paddle_x, paddle_y = sample_init_state(
            key, self.rows, self.columns
        )
        # Last two state vector correspond to timestep and done
        state = EnvState(
            ball_x=ball_x,
            ball_y=ball_y,
            paddle_x=paddle_x,
            paddle_y=paddle_y,
            prev_done=False,
            time=jnp.int32(0),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((self.rows, self.columns))
        obs = obs.at[state.ball_y, state.ball_x].set(1.0)
        obs = obs.at[state.paddle_y, state.paddle_x].set(1.0)
        return obs

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return state.ball_y == self.rows - 1

    @property
    def name(self) -> str:
        """Environment name."""
        return "Catch-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.rows, self.columns), dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "ball_x": spaces.Discrete(self.columns),
                "ball_y": spaces.Discrete(self.rows),
                "paddle_x": spaces.Discrete(self.columns),
                "paddle_y": spaces.Discrete(self.rows),
                "prev_done": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def render(self, state: EnvState, _: Any):  # params: EnvParams):
        """Small utility for plotting the agent's state."""

        fig, ax = plt.subplots()
        ax.imshow(jnp.zeros((self.rows, self.columns)), cmap="Greys", vmin=0, vmax=1)
        ax.annotate(
            "P",
            fontsize=20,
            xy=(state.paddle_x, state.paddle_y),
            xycoords="data",
            xytext=(state.paddle_x - 0.3, state.paddle_y + 0.25),
        )
        ax.annotate(
            "B",
            fontsize=20,
            xy=(state.ball_x, state.ball_y),
            xycoords="data",
            xytext=(state.ball_x - 0.3, state.ball_y + 0.25),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


def sample_init_state(
    key: jax.Array, rows: int, columns: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample a new initial state."""
    ball_x = jax.random.randint(
        key, shape=(), minval=0, maxval=columns, dtype=jnp.int32
    )
    ball_y = jnp.int32(0)
    paddle_x = jnp.int32(columns // 2)
    paddle_y = jnp.int32(rows - 1)
    return ball_x, ball_y, paddle_x, paddle_y

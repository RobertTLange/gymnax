"""JAX Compatible version of Catch bsuite environment.


Source: github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py.
"""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    ball_x: chex.Array
    ball_y: chex.Array
    paddle_x: int
    paddle_y: int
    prev_done: bool
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 1000


class Catch(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of Catch bsuite environment."""

    def __init__(self, rows: int = 10, columns: int = 5):
        super().__init__()
        self.rows = rows
        self.columns = columns

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
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
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
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
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((self.rows, self.columns))
        obs = obs.at[state.ball_y, state.ball_x].set(1.0)
        obs = obs.at[state.paddle_y, state.paddle_x].set(1.0)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_loose = state.ball_y == self.rows - 1
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_loose, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Catch-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.rows, self.columns), dtype=jnp.int_)

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
    key: chex.PRNGKey, rows: int, columns: int
) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
    """Sample a new initial state."""
    ball_x = jax.random.randint(key, shape=(), minval=0, maxval=columns)
    ball_y = 0
    paddle_x = columns // 2
    paddle_y = rows - 1
    return ball_x, jnp.array(ball_y), paddle_x, paddle_y

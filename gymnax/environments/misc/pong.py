"""JAX compatible version of Pong-like environment.


Adapted from:
https://github.com/BlackHC/batch_pong_poc/blob/master/src/vanilla_pong.py -
Actions are encoded as: ['n', 'u', 'd']
"""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    paddle_centers: chex.Array
    ball_position: chex.Array
    last_ball_position: chex.Array
    ball_velocity: chex.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    ball_max_y_speed: int = 3
    paddle_y_speed: int = 1
    ball_x_speed: int = 1
    use_ai_policy: bool = True
    max_steps_in_episode: int = 1000


class Pong(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of Pong-like environment."""

    def __init__(
        self,
        width: int = 40,
        height: int = 30,
        paddle_half_height: int = 2,
    ):
        super().__init__()
        # 3 channels: P1/P2 and ball_t, ball_t-1
        self.obs_shape = (height, width, 3)
        self.action_set = jnp.array([0, 1, 2])
        self.width = width
        self.height = height
        self.paddle_half_height = paddle_half_height

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
        last_ball_position = state.ball_position

        state = move_paddles(
            action,
            params.paddle_y_speed,
            state,
            self.paddle_half_height,
            self.height,
            params.use_ai_policy,
        )
        state = move_ball(state)
        state = reflect_on_borders(state, self.height)
        state = reflect_on_paddle(state, self.width, self.paddle_half_height, params)

        # Check game condition & no. steps for termination condition
        state = state.replace(
            last_ball_position=last_ball_position, time=state.time + 1
        )
        done = self.is_terminal(state, params)

        reward = jnp.array(1.0 * (1 - done))
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        paddle_centers = jnp.array([self.height / 2, self.height / 2])
        ball_position = jnp.array([self.height / 2, self.width / 2])
        ball_velocity = jnp.array([0, params.ball_x_speed])
        state = EnvState(
            paddle_centers=paddle_centers,  # p1/p2
            ball_position=ball_position,  # row/col
            last_ball_position=ball_position,  # row/col
            ball_velocity=ball_velocity,  # x/y vel
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((self.height, self.width, 3))
        ball_index = jnp.floor(state.ball_position)
        h = jnp.clip(ball_index[0], 0, self.height - 1).astype(jnp.int32)
        w = jnp.clip(ball_index[1], 0, self.width - 1).astype(jnp.int32)

        ball_index_last = jnp.floor(state.last_ball_position)
        h_l = jnp.clip(ball_index_last[0], 0, self.height - 1).astype(jnp.int32)
        w_l = jnp.clip(ball_index_last[1], 0, self.width - 1).astype(jnp.int32)

        obs = obs.at[h, w, 1].set(1)  # ball
        obs = obs.at[h_l, w_l, 2].set(1)  # ball last time step

        paddle_range = jnp.arange(
            -self.paddle_half_height, self.paddle_half_height + 1
        )[jnp.newaxis, :]

        paddle_indices = jnp.floor(state.paddle_centers)
        expanded_paddles = jnp.clip(
            paddle_indices[:, jnp.newaxis] + paddle_range, 0, self.height - 1
        ).astype(jnp.int32)

        obs = obs.at[
            expanded_paddles, jnp.array([0, self.width - 1]).reshape((2, 1)), 0
        ].set(
            1
        )  # paddle
        return obs.reshape((self.height, self.width, 3)).astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        done_term = update_game_state(state, self.width)
        done = jnp.logical_or(jnp.array(done_steps), jnp.array(done_term))
        return jnp.logical_or(done, state.terminal)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pong-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 2, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "paddle_centers": spaces.Box(0, self.height, (2,)),
                "ball_position": spaces.Box(0, self.height, (2,)),
                "last_ball_position": spaces.Box(0, self.height, (2,)),
                "ball_velocity": spaces.Box(0, self.height, (2,)),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )

    def render(self, state: EnvState, _: EnvParams):
        """Small utility for plotting the agent's state."""

        _, ax = plt.subplots()
        obs = self.get_obs(state)
        n_channels = self.obs_shape[-1]
        # The seaborn color_palette cubhelix is used to
        # assign visually distinct colors to each channel for the env
        cmap = sns.color_palette("cubehelix", n_channels)
        cmap.insert(0, (0, 0, 0))
        cmap = colors.ListedColormap(cmap)
        bounds = [i for i in range(n_channels + 2)]
        norm = colors.BoundaryNorm(bounds, n_channels + 1)
        numerical_state = (
            jnp.amax(obs * jnp.reshape(jnp.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation="none")


def update_game_state(state: EnvState, width: int) -> jnp.ndarray:
    """Check if right or left border win conditions are met."""
    win_right = state.ball_position[1] < 0
    win_left = state.ball_position[1] >= width
    return jnp.logical_or(win_right, win_left)


def reflect_on_borders(state: EnvState, height: int) -> EnvState:
    """Reflect ball at the bottom/top border of the frame."""
    # these are not really used?
    #   reflect_bottom = state.ball_position[0] < 0
    #   ball_position = jax.lax.select(
    #       reflect_bottom,
    #       state.ball_position.at[0].set(state.ball_position[0] * -1),
    #       state.ball_position,
    #   )
    #   ball_velocity = jax.lax.select(
    #       reflect_bottom,
    #       state.ball_velocity.at[0].set(state.ball_velocity[0] * -1),
    #       state.ball_velocity,
    #   )

    reflect_top = state.ball_position[0] >= height
    ball_position = jax.lax.select(
        reflect_top,
        state.ball_position.at[0].set(2 * (height - 1) - state.ball_position[0]),
        state.ball_position,
    )
    ball_velocity = jax.lax.select(
        reflect_top,
        state.ball_velocity.at[0].set(state.ball_velocity[0] * -1),
        state.ball_velocity,
    )
    return state.replace(ball_position=ball_position, ball_velocity=ball_velocity)


def reflect_on_paddle(
    state: EnvState, width: int, paddle_half_height: int, env_params: EnvParams
):
    """Reflect ball on paddle contact."""
    left_paddle_reflected_x = 2 * 1 - state.ball_position[1]
    right_paddle_reflected_x = 2 * (width - 2) - state.ball_position[1]

    paddle_height_distance = state.ball_position[jnp.newaxis, 0] - state.paddle_centers

    left_paddle_hit = jnp.logical_and(
        left_paddle_reflected_x >= 1,
        jnp.fabs(paddle_height_distance[0]) <= paddle_half_height,
    )
    right_paddle_hit = jnp.logical_and(
        right_paddle_reflected_x < width - 2,
        jnp.fabs(paddle_height_distance[1]) < paddle_half_height + 1,
    )

    # Left paddle hit updates
    left_ball_position = state.ball_position.at[1].set(left_paddle_reflected_x)
    left_ball_velocity = state.ball_velocity.at[1].set(state.ball_velocity[1] * -1)
    left_ball_velocity = left_ball_velocity.at[0].set(
        jnp.clip(
            left_ball_velocity[0] + paddle_height_distance[0] / paddle_half_height,
            -env_params.ball_max_y_speed,
            env_params.ball_max_y_speed,
        )
    )
    ball_position = jax.lax.select(
        left_paddle_hit, left_ball_position, state.ball_position
    )
    ball_velocity = jax.lax.select(
        left_paddle_hit, left_ball_velocity, state.ball_velocity
    )

    # Right paddle hit updates
    right_ball_position = ball_position.at[1].set(right_paddle_reflected_x)
    right_ball_velocity = ball_velocity.at[1].set(ball_velocity[1] * -1)
    right_ball_velocity = right_ball_velocity.at[0].set(
        jnp.clip(
            right_ball_velocity[0] + paddle_height_distance[1] / paddle_half_height,
            -env_params.ball_max_y_speed,
            env_params.ball_max_y_speed,
        )
    )
    ball_position = jax.lax.select(right_paddle_hit, right_ball_position, ball_position)
    ball_velocity = jax.lax.select(right_paddle_hit, right_ball_velocity, ball_velocity)
    return state.replace(
        ball_position=ball_position,
        ball_velocity=ball_velocity,
    )


def move_ball(state: EnvState) -> EnvState:
    """Update ball position using velocity."""
    ball_position = state.ball_position + state.ball_velocity
    return state.replace(ball_position=ball_position)


def move_paddles(
    action: int,
    paddle_y_speed: int,
    state: EnvState,
    paddle_half_height: int,
    height: int,
    use_ai_policy: bool,
) -> EnvState:
    """Update paddle positions and clip at height borders."""
    paddle_direction = -1 * (action == 1) + 1 * (action == 2)
    paddle_step = paddle_direction * paddle_y_speed
    # NOTE: Different from reference - full paddle is visible
    # Calculate new center of P1 based on action
    new_center_p1 = jnp.clip(
        state.paddle_centers[0] + paddle_step,
        paddle_half_height,
        height - paddle_half_height - 1,
    )
    # Calculate new center of P2 based on same action
    # This means both players play 'same' policy
    new_center_self = jnp.clip(
        state.paddle_centers[1] + paddle_step,
        paddle_half_height,
        height - paddle_half_height - 1,
    )

    # Calculate new center of P2 based on 'AI' policy
    # Minimize distance to ball!
    dist_center_down = jnp.abs(
        state.ball_position[0]
        - jnp.clip(
            state.paddle_centers[1] + paddle_y_speed,
            paddle_half_height,
            height - paddle_half_height - 1,
        )
    )
    dist_center_up = jnp.abs(
        state.ball_position[0]
        - jnp.clip(
            state.paddle_centers[1] - paddle_y_speed,
            paddle_half_height,
            height - paddle_half_height - 1,
        )
    )
    ai_go_up = dist_center_up < dist_center_down
    new_center_ai = jnp.clip(
        state.paddle_centers[1]
        - ai_go_up * paddle_y_speed
        + (1 - ai_go_up) * paddle_y_speed,
        paddle_half_height,
        height - paddle_half_height - 1,
    )
    new_center_p2 = jax.lax.select(use_ai_policy, new_center_ai, new_center_self)

    new_centers = jnp.array([new_center_p1, new_center_p2])
    return state.replace(paddle_centers=new_centers)

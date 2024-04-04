"""JAX compatible version of Breakout MinAtar environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    ball_y: jnp.ndarray
    ball_x: jnp.ndarray
    ball_dir: jnp.ndarray
    pos: int
    brick_map: chex.Array
    strike: bool
    last_y: jnp.ndarray
    last_x: jnp.ndarray
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 1000


class MinBreakout(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of Breakout MinAtar environment.


    Source:
    github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py


    ENVIRONMENT DESCRIPTION - 'Breakout-MinAtar'
    - Player controls paddle on bottom of screen.
    - Must bounce ball to break 3 rows if bricks along top of screen.
    - A reward of +1 is given for each broken brick.
    - If all bricks are cleared another 3 rows are added.
    - Ball travels only along diagonals, when paddle/wall hit it bounces off
    - Termination if ball hits bottom of screen.
    - Ball direction is indicated by a trail channel.
    - There is no difficulty increase.
    - Channels are encoded as follows: 'paddle':0, 'ball':1, 'trail':2, 'brick':3
    - Observation has dimensionality (10, 10, 4)
    - Actions are encoded as follows: ['n','l','r']
    """

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 4)
        # Full action set: ['n','l','u','r','d','f']
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        # Minimal action set: ['n', 'l', 'r']
        self.minimal_action_set = jnp.array([0, 1, 3])
        # Set active action set for environment
        # If minimal map to integer in full action set
        if use_minimal_action_set:
            self.action_set = self.minimal_action_set
        else:
            self.action_set = self.full_action_set

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
        a = self.action_set[action]
        state, new_x, new_y = step_agent(state, a)
        state, reward = step_ball_brick(state, new_x, new_y)

        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
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
        ball_start = jax.random.choice(key, jnp.array([0, 1]), shape=())
        state = EnvState(
            ball_y=jnp.array(3),
            ball_x=jnp.array([0, 9])[ball_start],
            ball_dir=jnp.array([2, 3])[ball_start],
            pos=4,
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            last_y=jnp.array(3),
            last_x=jnp.array([0, 9])[ball_start],
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(self.obs_shape, dtype=bool)
        # Set the position of the player paddle, paddle, trail & brick map
        obs = obs.at[9, state.pos, 0].set(1)
        obs = obs.at[state.ball_y, state.ball_x, 1].set(1)
        obs = obs.at[state.last_y, state.last_x, 2].set(1)
        obs = obs.at[:, :, 3].set(state.brick_map)
        return obs.astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_steps, state.terminal)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Breakout-MinAtar"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "ball_y": spaces.Discrete(10),
                "ball_x": spaces.Discrete(10),
                "ball_dir": spaces.Discrete(10),
                "pos": spaces.Discrete(10),
                "brick_map": spaces.Box(0, 1, (10, 10)),
                "strike": spaces.Discrete(2),
                "last_y": spaces.Discrete(10),
                "last_x": spaces.Discrete(10),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(
    state: EnvState,
    action: jnp.ndarray,
) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Helper that steps the agent and checks boundary conditions."""
    # Update player position
    pos = (
        # Action left & border condition
        jnp.maximum(0, state.pos - 1) * (action == 1)
        # Action right & border condition
        + jnp.minimum(9, state.pos + 1) * (action == 3)
        # Don't move player if not l/r chosen
        + state.pos * jnp.logical_and(action != 1, action != 3)
    )

    # Update ball position - based on direction of movement
    last_x = state.ball_x
    last_y = state.ball_y
    new_x = (
        (state.ball_x - 1) * (state.ball_dir == 0)
        + (state.ball_x + 1) * (state.ball_dir == 1)
        + (state.ball_x + 1) * (state.ball_dir == 2)
        + (state.ball_x - 1) * (state.ball_dir == 3)
    )
    new_y = (
        (state.ball_y - 1) * (state.ball_dir == 0)
        + (state.ball_y - 1) * (state.ball_dir == 1)
        + (state.ball_y + 1) * (state.ball_dir == 2)
        + (state.ball_y + 1) * (state.ball_dir == 3)
    )

    # Boundary conditions for x position
    border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    new_x = jax.lax.select(border_cond_x, (0 * (new_x < 0) + 9 * (new_x > 9)), new_x)
    # Reflect ball direction if bounced off at x border
    ball_dir = jax.lax.select(
        border_cond_x, jnp.array([1, 0, 3, 2])[state.ball_dir], state.ball_dir
    )
    return (
        state.replace(
            pos=pos,
            last_x=last_x,
            last_y=last_y,
            ball_dir=ball_dir,
        ),
        new_x,
        new_y,
    )


def step_ball_brick(
    state: EnvState, new_x: jnp.ndarray, new_y: jnp.ndarray
) -> Tuple[EnvState, jnp.ndarray]:
    """Helper that computes reward and termination cond. from brickmap."""

    reward = 0

    # Reflect ball direction if bounced off at y border
    border_cond1_y = new_y < 0
    new_y = lax.select(border_cond1_y, 0, new_y)
    ball_dir = lax.select(
        border_cond1_y, jnp.array([3, 2, 1, 0])[state.ball_dir], state.ball_dir
    )

    # 1st NASTY ELIF BEGINS HERE... = Brick collision
    strike_toggle = jnp.logical_and(
        1 - border_cond1_y, state.brick_map[new_y, new_x] == 1
    )
    strike_bool = jnp.logical_and((1 - state.strike), strike_toggle)
    reward += strike_bool * 1.0
    # next line wasn't used anywhere
    # strike = jax.lax.select(strike_toggle, strike_bool, False)

    brick_map = jax.lax.select(
        strike_bool, state.brick_map.at[new_y, new_x].set(0), state.brick_map
    )
    new_y = jax.lax.select(strike_bool, state.last_y, new_y)
    ball_dir = jax.lax.select(strike_bool, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir)

    # 2nd NASTY ELIF BEGINS HERE... = Wall collision
    brick_cond = jnp.logical_and(1 - strike_toggle, new_y == 9)

    # Spawn new bricks if there are no more around - everything is collected
    spawn_bricks = jnp.logical_and(brick_cond, jnp.count_nonzero(brick_map) == 0)
    brick_map = jax.lax.select(spawn_bricks, brick_map.at[1:4, :].set(1), brick_map)

    # Redirect ball because it collided with old player position
    redirect_ball1 = jnp.logical_and(brick_cond, state.ball_x == state.pos)
    ball_dir = jax.lax.select(
        redirect_ball1, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball1, state.last_y, new_y)

    # Redirect ball because it collided with new player position
    redirect_ball2a = jnp.logical_and(brick_cond, 1 - redirect_ball1)
    redirect_ball2 = jnp.logical_and(redirect_ball2a, new_x == state.pos)
    ball_dir = jax.lax.select(
        redirect_ball2, jnp.array([2, 3, 0, 1])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball2, state.last_y, new_y)
    redirect_cond = jnp.logical_and(1 - redirect_ball1, 1 - redirect_ball2)
    terminal = jnp.logical_and(brick_cond, redirect_cond)

    strike = jax.lax.select(1 - strike_toggle == 1, False, True)
    return (
        state.replace(
            ball_dir=ball_dir,
            brick_map=brick_map,
            strike=strike,
            ball_x=new_x,
            ball_y=new_y,
            terminal=terminal,
        ),
        reward,
    )

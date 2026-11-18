"""JAX implementation of a 2D swimmer environment with urchin obstacles."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the Swimmer environment.

    Attributes:
        urchin_xys: Array of shape `(num_urchins, 2)` with the `(x, y)` position
            of each urchin.
        xy: Position of the swimmer as `(x, y)`.
        xy_vel: Velocity of the swimmer as `(x, y)`.
        goal_xy: Position of the goal as `(x, y)`.
        time: Number of steps taken in the current episode.
    """

    urchin_xys: jax.Array
    xy: jax.Array
    xy_vel: jax.Array
    goal_xy: jax.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the Swimmer environment.

    Attributes:
        dt: Duration of one step, used for Euler integration.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    dt: float = 0.05
    max_steps_in_episode: int = 500


class Swimmer(environment.Environment[EnvState, EnvParams]):
    """Swim to a goal position in the plane while keeping away from urchins.

    The action sets the acceleration of the swimmer, which is integrated with step
    size `params.dt`. At every reset the positions of the urchins, the swimmer and
    the goal are sampled uniformly in `[-1, 1]` for both coordinates and the
    swimmer starts at rest. Urchins and goal stay fixed for the whole episode.
    Supports differentiable transitions via `with_transition_gradients()`.

    - Observation: `Box` of shape `(2 * num_urchins + 6,)` with the urchin
      positions, the swimmer position, the swimmer velocity and the goal position.
    - Actions: `Box` of shape `(2,)` with bounds `[-1, 1]`. The acceleration of the
      swimmer in x and y.
    - Reward: `exp(-0.5 * d**2)`, where `d` is the distance between swimmer and
      goal after the step, multiplied by `1 - exp(-30 * u**2)` for every urchin
      at distance `u` from the swimmer.
    - Termination: never.
    - Truncation: after `params.max_steps_in_episode` steps (500 by default).

    Args:
        num_urchins: Number of urchins placed in the environment.

    References:
        - [ivy_gym implementation](https://github.com/unifyai/gym/blob/master/ivy_gym/swimmer.py)
    """

    _supports_transition_gradients = True

    def __init__(self, num_urchins: int = 5):
        super().__init__()
        self.num_urchins = num_urchins

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Swimmer-misc."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Integrate the swimmer dynamics and compute the goal and urchin reward."""
        xy_vel = state.xy_vel + params.dt * action
        xy = state.xy + params.dt * xy_vel

        state = EnvState(
            urchin_xys=state.urchin_xys,
            xy=xy,
            xy_vel=xy_vel,
            goal_xy=state.goal_xy,
            time=state.time + 1,
        )

        rew = jnp.exp(-0.5 * jnp.sum((state.xy - state.goal_xy) ** 2))
        # Urchins proximity.
        reward = rew * jnp.prod(
            1 - jnp.exp(-30 * jnp.sum((state.xy - state.urchin_xys) ** 2, axis=-1)),
            axis=-1,
        )

        done = self.is_terminal(state, params)
        observation, state = self._apply_transition_gradient_policy(
            self.get_obs(state, params), state
        )
        return (
            observation,
            state,
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling urchins, swimmer position and goal."""
        # Sample reward function + construct state as concat with timestamp
        key_urchin, key_xy, key_goal = jax.random.split(key, 3)

        urchin_xys = jax.random.uniform(
            key_urchin, minval=-1, maxval=1, shape=(self.num_urchins, 2)
        )
        xy = jax.random.uniform(key_xy, minval=-1, maxval=1, shape=(2,))
        xy_vel = jnp.zeros((2,))
        goal_xy = jax.random.uniform(key_goal, minval=-1, maxval=1, shape=(2,))

        state = EnvState(
            urchin_xys=urchin_xys,
            xy=xy,
            xy_vel=xy_vel,
            goal_xy=goal_xy,
            time=0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Concatenate urchin positions, swimmer position, velocity and goal."""
        ob = (
            jnp.reshape(state.urchin_xys, (-1, 2)),
            jnp.reshape(state.xy, (-1, 2)),
            jnp.reshape(state.xy_vel, (-1, 2)),
            jnp.reshape(state.goal_xy, (-1, 2)),
        )
        ob = jnp.concatenate(ob, axis=0)
        return jnp.reshape(ob, (-1,))

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Swimmer-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        # if params is None:
        #   params = self.default_params
        low = jnp.array(2 * [-1], dtype=jnp.float32)
        high = jnp.array(2 * [1], dtype=jnp.float32)
        return spaces.Box(low, high, (2,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            (6 + self.num_urchins * 2) * [-jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        high = jnp.array(
            (6 + self.num_urchins * 2) * [jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (6 + self.num_urchins * 2,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "urchin_xys": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (self.num_urchins,),
                    jnp.float32,
                ),
                "xy": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (2,),
                    jnp.float32,
                ),
                "xy_vel": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (2,),
                    jnp.float32,
                ),
                "goal_xy": spaces.Box(
                    -1,
                    1,
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

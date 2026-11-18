"""JAX implementation of the MountainCar-v0 Gymnasium environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the MountainCar environment.

    Attributes:
        position: Position of the car, clipped to
            `[params.min_position, params.max_position]`.
        velocity: Velocity of the car, clipped to
            `[-params.max_speed, params.max_speed]`.
        time: Number of steps taken in the current episode.
    """

    position: jax.Array
    velocity: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the MountainCar environment.

    Attributes:
        min_position: Lower bound to which the position is clipped. The velocity
            is set to 0 when the car reaches it while moving left.
        max_position: Upper bound to which the position is clipped.
        max_speed: Maximum absolute velocity of the car.
        goal_position: Position at or beyond which the goal is reached.
        goal_velocity: Minimum velocity required at the goal for termination.
        force: Change in velocity per step caused by accelerating left or right.
        gravity: Scale of the slope term `cos(3 * position)` that is subtracted
            from the velocity in every step.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    goal_velocity: float = 0.0
    force: float = 0.001
    gravity: float = 0.0025
    max_steps_in_episode: int = 200


class MountainCar(environment.Environment[EnvState, EnvParams]):
    """Drive an underpowered car up a hill by building momentum in a valley.

    JAX port of the Gymnasium `MountainCar-v0` environment. The car starts at a
    uniformly random position in `[-0.6, -0.4]` with zero velocity.

    - Observation: `Box` of shape `(2,)` with the position and velocity of the car.
    - Actions: `Discrete(3)`. 0 accelerates to the left, 1 does not accelerate and
      2 accelerates to the right.
    - Reward: -1 for every step, including the terminating one.
    - Termination: the position is at least `params.goal_position` and the
      velocity at least `params.goal_velocity`.
    - Truncation: after `params.max_steps_in_episode` steps (200 by default).

    References:
        - [Moore (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
        - [Gymnasium implementation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py)
    """

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for MountainCar-v0."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Perform single timestep state transition."""
        velocity = (
            state.velocity
            + (action - 1) * params.force
            - jnp.cos(3 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (1 - (position == params.min_position) * (velocity < 0))

        reward = -1.0

        # Update state dict and evaluate termination conditions
        state = EnvState(position=position, velocity=velocity, time=state.time + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=init_state, velocity=jnp.array(0.0), time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return the position and velocity of the car."""
        return jnp.array([state.position, state.velocity])

    def is_terminated(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done1 = (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )

        return done1

    @property
    def name(self) -> str:
        """Environment name."""
        return "MountainCar-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )

        return spaces.Dict(
            {
                "position": spaces.Box(low[0], high[0], (), dtype=jnp.float32),
                "velocity": spaces.Box(low[1], high[1], (), dtype=jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

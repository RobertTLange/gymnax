"""JAX implementation of the MountainCarContinuous-v0 Gymnasium environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the ContinuousMountainCar environment.

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
    """Parameters of the ContinuousMountainCar environment.

    Attributes:
        min_action: Lower bound to which the action is clipped.
        max_action: Upper bound to which the action is clipped.
        min_position: Lower bound to which the position is clipped.
        max_position: Upper bound to which the position is clipped.
        max_speed: Maximum absolute velocity of the car.
        goal_position: Position at or beyond which the goal is reached. A negative
            velocity is set to 0 while the car is at or beyond it.
        goal_velocity: Minimum velocity required at the goal for termination.
        power: Change in velocity per step caused by a unit of clipped action.
        gravity: Scale of the slope term `cos(3 * position)` that is subtracted
            from the velocity in every step.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    min_action: float = -1.0
    max_action: float = 1.0
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    power: float = 0.0015
    gravity: float = 0.0025
    max_steps_in_episode: int = 999


class ContinuousMountainCar(environment.Environment[EnvState, EnvParams]):
    """Drive an underpowered car up a hill using a continuous force.

    JAX port of the Gymnasium `MountainCarContinuous-v0` environment. The car
    starts at a uniformly random position in `[-0.6, -0.4]` with zero velocity.
    Supports differentiable transitions via `with_transition_gradients()`.

    - Observation: `Box` of shape `(2,)` with the position and velocity of the car.
    - Actions: `Box` of shape `(1,)` with the force on the car, bounded by
      `params.min_action` and `params.max_action` (-1 and 1 by default). Positive
      values accelerate to the right; actions outside the bounds are clipped.
    - Reward: `-0.1 * action**2` for every step, computed from the action before
      clipping, plus 100 for the step that reaches the goal.
    - Termination: the position is at least `params.goal_position` and the
      velocity at least `params.goal_velocity`.
    - Truncation: after `params.max_steps_in_episode` steps (999 by default).

    References:
        - [Moore (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
        - [Gymnasium implementation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/continuous_mountain_car.py)
    """

    _supports_transition_gradients = True

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for MountainCarContinuous-v0."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        force = jnp.clip(action, params.min_action, params.max_action)
        velocity = (
            state.velocity
            + force * params.power
            - jnp.cos(3 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (1 - (position >= params.goal_position) * (velocity < 0))

        reward = -0.1 * action**2 + 100 * (
            (position >= params.goal_position) * (velocity >= params.goal_velocity)
        )
        reward = reward.squeeze()

        # Update state dict and evaluate termination conditions
        state = EnvState(
            position=position.squeeze(),
            velocity=velocity.squeeze(),
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        observation, state = self._apply_transition_gradient_policy(
            self.get_obs(state), state
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
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=init_state, velocity=jnp.array(0.0), time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return the position and velocity of the car."""
        return jnp.array([state.position, state.velocity]).squeeze()

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        done1 = (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )
        return done1.squeeze()

    @property
    def name(self) -> str:
        """Environment name."""
        return "MountainCarContinuous-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(1,),
        )

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
        return spaces.Box(low, high, shape=(2,), dtype=jnp.float32)

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

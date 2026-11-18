"""JAX implementation of the FrozenLake Gymnasium environment."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces

START, FROZEN, HOLE, GOAL = range(4)
_TILES = {"S": START, "F": FROZEN, "H": HOLE, "G": GOAL}
_MAPS = {
    "4x4": ("SFFF", "FHFH", "FFFH", "HFFG"),
    "8x8": (
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
}
_DIRECTIONS = jnp.array(((0, -1), (1, 0), (0, 1), (-1, 0)))


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the FrozenLake environment.

    Attributes:
        position: Row-major index of the agent's tile, `row * ncol + col`.
        time: Number of steps taken in the current episode.
    """

    position: jax.Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the FrozenLake environment.

    Attributes:
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    max_steps_in_episode: int = 100


class FrozenLake(environment.Environment[EnvState, EnvParams]):
    """Cross a frozen lake from the start tile to the goal without falling in.

    JAX port of the Gymnasium `FrozenLake` environment with compatible maps,
    dynamics and rewards. The map is a grid of start (`S`), frozen (`F`), hole
    (`H`) and goal (`G`) tiles, and every episode starts on the start tile. Moves
    off the grid leave the position unchanged. If `is_slippery`, the agent moves
    in the chosen direction with probability `success_rate` and otherwise in one
    of the two perpendicular directions with equal probability. The probability
    of the executed move is returned as `info["p"]`.

    - Observation: `Discrete(nrow * ncol)`. The row-major index of the agent's
      tile, `row * ncol + col`.
    - Actions: `Discrete(4)`. 0 moves left, 1 down, 2 right and 3 up.
    - Reward: given by `reward_schedule` for the tile the agent lands on. By
      default 1 for the goal and 0 for all other tiles.
    - Termination: the agent lands on a hole or the goal.
    - Truncation: after `params.max_steps_in_episode` steps (by default 200 for
      8 x 8 maps and 100 otherwise).

    Args:
        desc: Custom map as a sequence of equally long row strings made of the
            tiles `S`, `F`, `H` and `G`, with exactly one `S` and one `G`. If
            `None`, the map is selected by `map_name`.
        map_name: Name of the built-in map, `"4x4"` or `"8x8"`. Only used if
            `desc` is `None`.
        is_slippery: Whether moves are stochastic.
        success_rate: Probability of moving in the chosen direction if
            `is_slippery`. Must be in `[0, 1]`.
        reward_schedule: Rewards for landing on a goal, a hole and any other
            tile, in that order.

    References:
        - [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
    """

    def __init__(
        self,
        desc: Sequence[str] | None = None,
        map_name: str | None = "4x4",
        is_slippery: bool = True,
        success_rate: float = 1.0 / 3.0,
        reward_schedule: Sequence[float] = (1.0, 0.0, 0.0),
    ):
        if desc is None:
            if map_name not in _MAPS:
                raise ValueError("map_name must be '4x4' or '8x8' when desc is omitted")
            desc = _MAPS[map_name]

        self.desc = _map_from_strings(desc)
        self.nrow, self.ncol = self.desc.shape
        self.n_states = self.nrow * self.ncol
        self.is_slippery = is_slippery
        self.success_rate = _validate_success_rate(success_rate)
        self.reward_schedule = _validate_reward_schedule(reward_schedule)

    @property
    def default_params(self) -> EnvParams:
        """Default parameters with a limit of 200 steps for 8x8 maps, else 100."""
        max_steps = 200 if self.desc.shape == (8, 8) else 100
        return EnvParams(max_steps_in_episode=max_steps)

    @property
    def name(self) -> str:
        """Environment name."""
        return "FrozenLake-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Advance one transition before the base autoreset operation."""
        action, probability = self._transition_action(key, action)
        row, col = divmod(state.position, self.ncol)
        direction = _DIRECTIONS[action]
        row = jnp.clip(row + direction[0], 0, self.nrow - 1)
        col = jnp.clip(col + direction[1], 0, self.ncol - 1)
        position = row * self.ncol + col
        tile = self.desc[row, col]
        next_state = EnvState(position=position, time=state.time + 1)
        terminated = jnp.logical_or(tile == HOLE, tile == GOAL)
        reward = _reward_for_tile(tile, self.reward_schedule)
        observation = self.observe(key, next_state, action, params)
        return observation, next_state, reward, terminated, {"p": probability}

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the agent to the map's start tile."""
        start = jnp.argmax(self.desc.reshape(-1) == START)
        state = EnvState(position=start, time=0)
        return self.observe(key, state, None, params), state

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Return whether the agent reached a hole or goal."""
        del params
        return jnp.isin(self.desc.reshape(-1)[state.position], jnp.array((HOLE, GOAL)))

    def observe(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array | None,
        params: EnvParams,
    ) -> jax.Array:
        """Return the scalar row-major agent position."""
        del key, action, params
        return state.position

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Return the four-direction discrete action space."""
        del params
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the scalar discrete position space."""
        del params
        return spaces.Discrete(self.n_states)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """Return the internal position and elapsed-time spaces."""
        return spaces.Dict(
            {
                "position": spaces.Discrete(self.n_states),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )

    def _transition_action(
        self, key: jax.Array, action: int | float | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Sample the executed action and return it with its probability."""
        action = jnp.asarray(action, dtype=jnp.int32)
        if not self.is_slippery:
            return action, jnp.array(1.0, dtype=jnp.float32)

        key_direction, key_success = jax.random.split(key)
        offset = jax.random.choice(key_direction, jnp.array((-1, 1)))
        selected_action = jnp.where(
            jax.random.uniform(key_success) < self.success_rate,
            action,
            (action + offset) % self.num_actions,
        )
        return selected_action, jnp.where(
            selected_action == action,
            self.success_rate,
            (1.0 - self.success_rate) / 2.0,
        )


def _map_from_strings(desc: Sequence[str]) -> jax.Array:
    """Validate and convert a static character map to numeric tile IDs."""
    rows = tuple(desc)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("desc must be a non-empty rectangular map")
    if any(tile not in _TILES for row in rows for tile in row):
        raise ValueError("desc may only contain S, F, H, and G tiles")
    if sum(tile == "S" for row in rows for tile in row) != 1:
        raise ValueError("desc must contain exactly one start tile")
    if sum(tile == "G" for row in rows for tile in row) != 1:
        raise ValueError("desc must contain exactly one goal tile")
    return jnp.array([[_TILES[tile] for tile in row] for row in rows], dtype=jnp.int32)


def _validate_success_rate(success_rate: float) -> float:
    """Validate the intended-move probability."""
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate must be between 0 and 1")
    return float(success_rate)


def _validate_reward_schedule(reward_schedule: Sequence[float]) -> jax.Array:
    """Validate the goal, hole, and frozen reward values."""
    if len(reward_schedule) != 3:
        raise ValueError("reward_schedule must contain goal, hole, and frozen rewards")
    return jnp.asarray(reward_schedule, dtype=jnp.float32)


def _reward_for_tile(tile: jax.Array, reward_schedule: jax.Array) -> jax.Array:
    """Map a landing tile to its configured reward."""
    reward_index = jnp.where(tile == GOAL, 0, jnp.where(tile == HOLE, 1, 2))
    return reward_schedule[reward_index]

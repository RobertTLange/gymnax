"""JAX implementation of the MinAtar Seaquest environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces
from gymnax.environments.minatar.seaquest_helpers import (
    BOARD_SIZE,
    MAX_ENTITIES,
    active_mask,
    append,
    append_many,
    collide_bullets,
    compact,
    draw_entities,
    move_bullets,
    move_entities,
)


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the Seaquest environment.

    Entities are stored in buffers with `MAX_ENTITIES` rows. Only the leading rows of
    a buffer are active; their number is stored in the matching count field. In
    every buffer, `direction` is nonzero for entities moving right and zero for
    entities moving left.

    Attributes:
        oxygen: Remaining oxygen of the player's submarine.
        sub_x: Column of the player's submarine.
        sub_y: Row of the player's submarine; row 0 is the surface.
        sub_or: Orientation of the player's submarine: `True` facing right, `False`
            facing left.
        f_bullet_count: Number of active rows in `f_bullets`.
        f_bullets: Bullets of the player, with columns `[x, y, direction]`.
        e_bullet_count: Number of active rows in `e_bullets`.
        e_bullets: Bullets of the enemy submarines, with columns `[x, y, direction]`.
        e_fish_count: Number of active rows in `e_fish`.
        e_fish: Enemy fish, with columns `[x, y, direction, move_timer]`.
        e_subs_count: Number of active rows in `e_subs`.
        e_subs: Enemy submarines, with columns
            `[x, y, direction, move_timer, shot_timer]`.
        diver_count: Counter of divers carried by the player's submarine, at most 6.
            Set to -1, not 0, after surfacing with six divers; from then on it is
            one less than the number of divers actually carried.
        divers_count: Number of active rows in `divers`.
        divers: Divers in the water, with columns `[x, y, direction, move_timer]`.
        e_spawn_speed: Current number of steps between enemy spawns.
        e_spawn_timer: Steps left until the next enemy spawns.
        d_spawn_timer: Steps left until the next diver spawns.
        move_speed: Current move interval of the enemies; a fish or enemy submarine
            moves one cell every `move_speed + 1` steps.
        ramp_index: Number of difficulty increases so far.
        shot_timer: Steps left until the player can fire again.
        surface: Whether the player's submarine was at the surface after the
            previous step.
        time: Number of steps taken in the current episode.
        terminal: Whether the episode has terminated.
    """

    oxygen: jax.Array
    sub_x: jax.Array
    sub_y: jax.Array
    sub_or: jax.Array
    f_bullet_count: jax.Array
    f_bullets: jax.Array
    e_bullet_count: jax.Array
    e_bullets: jax.Array
    e_fish_count: jax.Array
    e_fish: jax.Array
    e_subs_count: jax.Array
    e_subs: jax.Array
    diver_count: jax.Array
    divers_count: jax.Array
    divers: jax.Array
    e_spawn_speed: jax.Array
    e_spawn_timer: jax.Array
    d_spawn_timer: jax.Array
    move_speed: jax.Array
    ramp_index: jax.Array
    shot_timer: jax.Array
    surface: jax.Array
    time: jax.Array
    terminal: jax.Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the Seaquest environment.

    Attributes:
        ramping: Whether the difficulty increases each time the player surfaces.
        ramp_interval: Unused.
        init_spawn_speed: Initial number of steps between enemy spawns.
        init_move_interval: Initial move interval of the enemies; a fish or enemy
            submarine moves one cell every `init_move_interval + 1` steps.
        max_oxygen: Oxygen of the player's submarine after a reset or after
            surfacing.
        diver_spawn_speed: Number of steps between diver spawns.
        shot_cool_down: Number of steps between two shots of the player.
        enemy_shot_interval: Shot interval of the enemy submarines; each of them
            shoots every `enemy_shot_interval + 1` steps.
        enemy_move_interval: Unused.
        diver_move_interval: Move interval of the divers; a diver moves one cell
            every `diver_move_interval + 1` steps.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    ramping: bool = True
    ramp_interval: int = 100
    init_spawn_speed: int = 20
    init_move_interval: int = 5
    max_oxygen: int = 200
    diver_spawn_speed: int = 30
    shot_cool_down: int = 5
    enemy_shot_interval: int = 10
    enemy_move_interval: int = 5
    diver_move_interval: int = 5
    max_steps_in_episode: int = 1000


def _hits_sub(entities: jax.Array, count: jax.Array, state: EnvState) -> jax.Array:
    """Check whether any active entity is at the position of the player's sub."""
    active = active_mask(count)
    at_sub = (entities[:, 0] == state.sub_x) & (entities[:, 1] == state.sub_y)
    return jnp.any(active & at_sub)


def _pick_up_divers(
    active: jax.Array, at_sub: jax.Array, diver_count: jax.Array
) -> jax.Array:
    """Select at most the remaining diver capacity in stable buffer order."""
    candidates = active & at_sub
    rank = jnp.cumsum(candidates.astype(jnp.int32)) - 1
    return candidates & (rank < jnp.maximum(6 - diver_count, 0))


class MinSeaquest(environment.Environment[EnvState, EnvParams]):
    """Rescue divers with a submarine while fighting off fish and enemy submarines.

    JAX port of the MinAtar Seaquest environment. The player controls a submarine in
    a 10x10 grid whose top row is the surface. It moves in rows 0 to 8 and fires
    bullets in the direction it is facing. Enemy fish and submarines spawn at the
    left and right edges of rows 1 to 8 and cross the grid horizontally; enemy
    submarines also shoot. Divers spawn in the same way and are picked up on contact,
    up to six at a time. The submarine uses one unit of oxygen for every step below
    the surface. Surfacing refills the oxygen and removes one carried diver, or all
    of them for a reward when six are carried. If `params.ramping` is set, the
    difficulty increases each time the player surfaces: enemies spawn and move
    faster.

    The transition order and the ten observation channels follow the MinAtar
    reference implementation. Entity lists are represented by fixed-capacity arrays
    so that the environment remains compatible with `jax.jit` and `jax.vmap`.

    - Observation: `Box` of shape `(10, 10, 10)` with binary channels: 0 player
      submarine, 1 cell behind the player submarine (indicating its orientation), 2
      friendly bullet, 3 trail (the cell behind each fish, enemy submarine and diver,
      indicating its direction), 4 enemy bullet, 5 enemy fish, 6 enemy submarine, 7
      oxygen gauge, 8 diver gauge, 9 diver. Both gauges are drawn in the bottom row:
      the oxygen gauge fills `oxygen * 10 // params.max_oxygen` cells from the left
      and the diver gauge marks one cell for each carried diver.
    - Actions: `Discrete(6)`: 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire. The
      minimal and the full action set are identical.
    - Reward: +1 for each enemy fish or submarine shot, and
      `oxygen * 10 // params.max_oxygen` for surfacing with six divers.
    - Termination: the player's submarine is hit by an enemy fish, submarine or
      bullet, runs out of oxygen, or surfaces while `state.diver_count` is 0.
    - Truncation: after `params.max_steps_in_episode` steps (1000 by default).

    Args:
        use_minimal_action_set: Whether to use the minimal instead of the full
            action set. Has no effect in Seaquest, since both sets contain all six
            actions.

    References:
        - [Young & Tian (2019)](https://arxiv.org/abs/1903.03176)
        - [MinAtar implementation](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/seaquest.py)
    """

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (BOARD_SIZE, BOARD_SIZE, 10)
        self.full_action_set = jnp.arange(6)
        self.minimal_action_set = jnp.arange(6)
        self.action_set = (
            self.minimal_action_set if use_minimal_action_set else self.full_action_set
        )

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Seaquest-MinAtar."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Apply one Seaquest frame in the MinAtar reference order."""
        key_enemy, key_diver = jax.random.split(key)
        state = self._spawn_entities(key_enemy, key_diver, state, params)
        state = self._step_agent(state, self.action_set[action], params)
        state, reward = self._step_friendly_bullets(state)
        state = self._step_divers(state, params)
        state, sub_reward, sub_terminal = self._step_enemy_subs(state, params)
        state, bullet_terminal = self._step_enemy_bullets(state)
        state, fish_reward, fish_terminal = self._step_enemy_fish(state)
        state, timer_reward, timer_terminal = self._step_timers(state, params)
        state = state.replace(time=state.time + 1)
        done = (
            self.is_terminal(state, params)
            | sub_terminal
            | bullet_terminal
            | fish_terminal
            | timer_terminal
        )
        state = state.replace(terminal=done)
        reward = reward + sub_reward + fish_reward + timer_reward
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            {"discount": self.discount(state, params)},
        )

    def _spawn_entities(
        self,
        key_enemy: jax.Array,
        key_diver: jax.Array,
        state: EnvState,
        params: EnvParams,
    ) -> EnvState:
        """Spawn an enemy and a diver when their spawn timers have run out."""
        enemy_choice_key, enemy_y_key = jax.random.split(key_enemy)
        enemy_values = jax.random.uniform(enemy_choice_key, shape=(2,))
        enemy_direction = enemy_values[0] < 0.5
        enemy_is_sub = enemy_values[1] < (1.0 / 3.0)
        enemy_y = jax.random.randint(enemy_y_key, (), 1, 9)
        enemy_x = jnp.where(enemy_direction, 0, 9)
        opposing_sub = (
            active_mask(state.e_subs_count)
            & (state.e_subs[:, 1] == enemy_y)
            & (state.e_subs[:, 2] != enemy_direction)
        )
        opposing_fish = (
            active_mask(state.e_fish_count)
            & (state.e_fish[:, 1] == enemy_y)
            & (state.e_fish[:, 2] != enemy_direction)
        )
        spawning_enemy = state.e_spawn_timer == 0
        can_spawn_enemy = spawning_enemy & ~jnp.any(opposing_sub | opposing_fish)
        enemy = jnp.array(
            [
                enemy_x,
                enemy_y,
                enemy_direction,
                state.move_speed,
                params.enemy_shot_interval,
            ]
        )
        fish = enemy[:4]
        subs, sub_count = append(
            state.e_subs, state.e_subs_count, enemy, can_spawn_enemy & enemy_is_sub
        )
        fishs, fish_count = append(
            state.e_fish, state.e_fish_count, fish, can_spawn_enemy & ~enemy_is_sub
        )

        diver_direction_key, diver_y_key = jax.random.split(key_diver)
        diver_direction = jax.random.bernoulli(diver_direction_key)
        diver_y = jax.random.randint(diver_y_key, (), 1, 9)
        diver = jnp.array(
            [
                jnp.where(diver_direction, 0, 9),
                diver_y,
                diver_direction,
                params.diver_move_interval,
            ]
        )
        divers, divers_count = append(
            state.divers, state.divers_count, diver, state.d_spawn_timer == 0
        )
        return state.replace(
            e_subs=subs,
            e_subs_count=sub_count,
            e_fish=fishs,
            e_fish_count=fish_count,
            divers=divers,
            divers_count=divers_count,
            e_spawn_timer=jnp.where(
                spawning_enemy, state.e_spawn_speed, state.e_spawn_timer
            ),
            d_spawn_timer=jnp.where(
                state.d_spawn_timer == 0, params.diver_spawn_speed, state.d_spawn_timer
            ),
        )

    def _step_agent(
        self, state: EnvState, action: jax.Array, params: EnvParams
    ) -> EnvState:
        """Move the player's submarine and fire a bullet if requested and possible."""
        is_left, is_up, is_right, is_down, is_fire = (
            action == 1,
            action == 2,
            action == 3,
            action == 4,
            action == 5,
        )
        sub_x = jnp.where(is_left, jnp.maximum(0, state.sub_x - 1), state.sub_x)
        sub_x = jnp.where(is_right, jnp.minimum(9, sub_x + 1), sub_x)
        sub_y = jnp.where(is_up, jnp.maximum(0, state.sub_y - 1), state.sub_y)
        sub_y = jnp.where(is_down, jnp.minimum(8, sub_y + 1), sub_y)
        sub_or = jnp.where(is_left, False, jnp.where(is_right, True, state.sub_or))
        can_fire = is_fire & (state.shot_timer == 0)
        bullet = jnp.array([sub_x, sub_y, sub_or])
        bullets, bullet_count = append(
            state.f_bullets, state.f_bullet_count, bullet, can_fire
        )
        return state.replace(
            sub_x=sub_x,
            sub_y=sub_y,
            sub_or=sub_or,
            f_bullets=bullets,
            f_bullet_count=bullet_count,
            shot_timer=jnp.where(can_fire, params.shot_cool_down, state.shot_timer),
        )

    def _step_friendly_bullets(self, state: EnvState) -> tuple[EnvState, jax.Array]:
        """Move the player's bullets and resolve their hits on fish and subs."""
        bullets, bullet_count = move_bullets(state.f_bullets, state.f_bullet_count)
        bullets, bullet_count, fish, fish_count, fish_reward = collide_bullets(
            bullets, bullet_count, state.e_fish, state.e_fish_count
        )
        bullets, bullet_count, subs, sub_count, sub_reward = collide_bullets(
            bullets, bullet_count, state.e_subs, state.e_subs_count
        )
        return state.replace(
            f_bullets=bullets,
            f_bullet_count=bullet_count,
            e_fish=fish,
            e_fish_count=fish_count,
            e_subs=subs,
            e_subs_count=sub_count,
        ), fish_reward + sub_reward

    def _step_divers(self, state: EnvState, params: EnvParams) -> EnvState:
        """Move the divers and pick up those at the player's sub, before and after."""
        # Diver capacity is deliberately separate from the fixed entity-buffer count.
        active = active_mask(state.divers_count)
        at_sub = (state.divers[:, 0] == state.sub_x) & (
            state.divers[:, 1] == state.sub_y
        )
        pick_up = _pick_up_divers(active, at_sub, state.diver_count)
        divers, divers_count = compact(state.divers, active & ~pick_up)
        rescued = state.diver_count + jnp.sum(pick_up, dtype=jnp.int32)
        divers, divers_count = move_entities(
            divers, divers_count, params.diver_move_interval
        )
        active = active_mask(divers_count)
        at_sub = (divers[:, 0] == state.sub_x) & (divers[:, 1] == state.sub_y)
        pick_up = _pick_up_divers(active, at_sub, rescued)
        divers, divers_count = compact(divers, active & ~pick_up)
        return state.replace(
            divers=divers,
            divers_count=divers_count,
            diver_count=rescue_count(rescued, pick_up),
        )

    def _step_enemy_subs(
        self, state: EnvState, params: EnvParams
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        """Move the enemy subs, let them shoot and check hits and collisions."""
        terminal_before = _hits_sub(state.e_subs, state.e_subs_count, state)
        subs, sub_count = move_entities(
            state.e_subs, state.e_subs_count, state.move_speed
        )
        terminal_after = _hits_sub(subs, sub_count, state)
        bullets, bullet_count, subs, sub_count, reward = collide_bullets(
            state.f_bullets, state.f_bullet_count, subs, sub_count
        )
        active = active_mask(sub_count)
        shoots = active & (subs[:, 4] == 0)
        next_shot_timer = jnp.where(
            shoots, params.enemy_shot_interval, jnp.maximum(subs[:, 4] - 1, 0)
        )
        subs = subs.at[:, 4].set(next_shot_timer)
        shots = subs[:, :3]
        enemy_bullets, enemy_bullet_count = append_many(
            state.e_bullets, state.e_bullet_count, shots, shoots
        )
        return (
            state.replace(
                f_bullets=bullets,
                f_bullet_count=bullet_count,
                e_subs=subs,
                e_subs_count=sub_count,
                e_bullets=enemy_bullets,
                e_bullet_count=enemy_bullet_count,
            ),
            reward,
            terminal_before | terminal_after,
        )

    def _step_enemy_bullets(self, state: EnvState) -> tuple[EnvState, jax.Array]:
        """Move the enemy bullets and check whether one hits the player's sub."""
        terminal_before = _hits_sub(state.e_bullets, state.e_bullet_count, state)
        bullets, bullet_count = move_bullets(state.e_bullets, state.e_bullet_count)
        return state.replace(
            e_bullets=bullets, e_bullet_count=bullet_count
        ), terminal_before | _hits_sub(bullets, bullet_count, state)

    def _step_enemy_fish(
        self, state: EnvState
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        """Move the enemy fish and check bullet hits and collisions with the sub."""
        terminal_before = _hits_sub(state.e_fish, state.e_fish_count, state)
        fish, fish_count = move_entities(
            state.e_fish, state.e_fish_count, state.move_speed
        )
        terminal_after = _hits_sub(fish, fish_count, state)
        bullets, bullet_count, fish, fish_count, reward = collide_bullets(
            state.f_bullets, state.f_bullet_count, fish, fish_count
        )
        return (
            state.replace(
                f_bullets=bullets,
                f_bullet_count=bullet_count,
                e_fish=fish,
                e_fish_count=fish_count,
            ),
            reward,
            terminal_before | terminal_after,
        )

    def _step_timers(
        self, state: EnvState, params: EnvParams
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        """Update timers and oxygen, and handle surfacing and difficulty ramping."""
        state = state.replace(
            e_spawn_timer=jnp.maximum(state.e_spawn_timer - 1, 0),
            d_spawn_timer=jnp.maximum(state.d_spawn_timer - 1, 0),
            shot_timer=jnp.maximum(state.shot_timer - 1, 0),
        )
        oxygen_empty = state.oxygen <= 0
        underwater = state.sub_y > 0
        oxygen = jnp.where(underwater, state.oxygen - 1, state.oxygen)
        surfacing = ~underwater & ~state.surface
        no_divers = surfacing & (state.diver_count == 0)
        full_rescue = state.diver_count == 6
        surface_reward = jnp.where(
            surfacing & full_rescue, state.oxygen * 10 // params.max_oxygen, 0
        )
        divers_after_surface = jnp.where(full_rescue, -1, state.diver_count - 1)
        diver_count = jnp.where(surfacing, divers_after_surface, state.diver_count)
        oxygen = jnp.where(surfacing, params.max_oxygen, oxygen)
        should_ramp = (
            params.ramping
            & surfacing
            & ((state.e_spawn_speed > 1) | (state.move_speed > 2))
        )
        move_speed = jnp.where(
            should_ramp & (state.move_speed > 2) & ((state.ramp_index % 2) == 1),
            state.move_speed - 1,
            state.move_speed,
        )
        spawn_speed = jnp.where(
            should_ramp & (state.e_spawn_speed > 1),
            state.e_spawn_speed - 1,
            state.e_spawn_speed,
        )
        ramp_index = state.ramp_index + should_ramp.astype(jnp.int32)
        return (
            state.replace(
                oxygen=oxygen,
                diver_count=diver_count,
                surface=jnp.where(
                    underwater, False, jnp.where(surfacing, True, state.surface)
                ),
                e_spawn_speed=spawn_speed,
                move_speed=move_speed,
                ramp_index=ramp_index,
            ),
            surface_reward.astype(jnp.float32),
            oxygen_empty | no_divers,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state to the fixed initial configuration."""
        del key
        state = EnvState(
            oxygen=jnp.array(params.max_oxygen, dtype=jnp.int32),
            sub_x=jnp.array(5, dtype=jnp.int32),
            sub_y=jnp.array(0, dtype=jnp.int32),
            sub_or=jnp.array(False),
            f_bullet_count=jnp.array(0, dtype=jnp.int32),
            f_bullets=jnp.zeros((MAX_ENTITIES, 3), dtype=jnp.int32),
            e_bullet_count=jnp.array(0, dtype=jnp.int32),
            e_bullets=jnp.zeros((MAX_ENTITIES, 3), dtype=jnp.int32),
            e_fish_count=jnp.array(0, dtype=jnp.int32),
            e_fish=jnp.zeros((MAX_ENTITIES, 4), dtype=jnp.int32),
            e_subs_count=jnp.array(0, dtype=jnp.int32),
            e_subs=jnp.zeros((MAX_ENTITIES, 5), dtype=jnp.int32),
            diver_count=jnp.array(0, dtype=jnp.int32),
            divers_count=jnp.array(0, dtype=jnp.int32),
            divers=jnp.zeros((MAX_ENTITIES, 4), dtype=jnp.int32),
            e_spawn_speed=jnp.array(params.init_spawn_speed, dtype=jnp.int32),
            e_spawn_timer=jnp.array(params.init_spawn_speed, dtype=jnp.int32),
            d_spawn_timer=jnp.array(params.diver_spawn_speed, dtype=jnp.int32),
            move_speed=jnp.array(params.init_move_interval, dtype=jnp.int32),
            ramp_index=jnp.array(0, dtype=jnp.int32),
            shot_timer=jnp.array(0, dtype=jnp.int32),
            surface=jnp.array(True),
            time=jnp.array(0, dtype=jnp.int32),
            terminal=jnp.array(False),
        )
        return self.get_obs(state, params), state

    def get_obs(
        self, state: EnvState, params: EnvParams | None = None, key=None
    ) -> jax.Array:
        """Encode fixed buffers into MinAtar's 10x10x10 observation."""
        del key
        if params is None:
            params = self.default_params
        obs = jnp.zeros(self.obs_shape, dtype=jnp.float32)
        obs = obs.at[state.sub_y, state.sub_x, 0].set(1)
        back_x = jnp.where(state.sub_or, state.sub_x - 1, state.sub_x + 1)
        obs = obs.at[state.sub_y, back_x, 1].set(1)
        oxygen_gauge = jnp.arange(BOARD_SIZE) < (
            jnp.maximum(state.oxygen, 0) * 10 // params.max_oxygen
        )
        diver_gauge = (jnp.arange(BOARD_SIZE) >= 9 - state.diver_count) & (
            jnp.arange(BOARD_SIZE) < 9
        )
        obs = obs.at[9, :, 7].set(oxygen_gauge).at[9, :, 8].set(diver_gauge)
        obs = draw_entities(obs, state.f_bullets, state.f_bullet_count, 2, False)
        obs = draw_entities(obs, state.e_bullets, state.e_bullet_count, 4, False)
        obs = draw_entities(obs, state.e_fish, state.e_fish_count, 5, True)
        obs = draw_entities(obs, state.e_subs, state.e_subs_count, 6, True)
        return draw_entities(obs, state.divers, state.divers_count, 9, True)

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return state.terminal

    @property
    def name(self) -> str:
        """Environment name."""
        return "Seaquest-MinAtar"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        del params
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        del params
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "oxygen": spaces.Discrete(params.max_oxygen + 1),
                "sub_x": spaces.Discrete(BOARD_SIZE),
                "sub_y": spaces.Discrete(9),
                "sub_or": spaces.Discrete(2),
                "f_bullet_count": spaces.Discrete(MAX_ENTITIES + 1),
                "f_bullets": spaces.Box(-1, BOARD_SIZE, (MAX_ENTITIES, 3)),
                "e_bullet_count": spaces.Discrete(MAX_ENTITIES + 1),
                "e_bullets": spaces.Box(-1, BOARD_SIZE, (MAX_ENTITIES, 3)),
                "e_fish_count": spaces.Discrete(MAX_ENTITIES + 1),
                "e_fish": spaces.Box(-1, BOARD_SIZE, (MAX_ENTITIES, 4)),
                "e_subs_count": spaces.Discrete(MAX_ENTITIES + 1),
                "e_subs": spaces.Box(-1, BOARD_SIZE, (MAX_ENTITIES, 5)),
                "diver_count": spaces.Box(-1, 6, ()),
                "divers_count": spaces.Discrete(MAX_ENTITIES + 1),
                "divers": spaces.Box(-1, BOARD_SIZE, (MAX_ENTITIES, 4)),
                "e_spawn_speed": spaces.Discrete(params.init_spawn_speed + 1),
                "e_spawn_timer": spaces.Discrete(params.init_spawn_speed + 1),
                "d_spawn_timer": spaces.Discrete(params.diver_spawn_speed + 1),
                "move_speed": spaces.Discrete(params.init_move_interval + 1),
                "ramp_index": spaces.Discrete(params.max_steps_in_episode + 1),
                "shot_timer": spaces.Discrete(params.shot_cool_down + 1),
                "surface": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
                "terminal": spaces.Discrete(2),
            }
        )


def rescue_count(rescued: jax.Array, picked_up: jax.Array) -> jax.Array:
    """Add the picked-up divers to the carried count, capped at six."""
    return jnp.minimum(6, rescued + jnp.sum(picked_up, dtype=jnp.int32))

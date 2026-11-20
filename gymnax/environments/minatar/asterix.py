"""JAX implementation of the MinAtar Asterix environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the Asterix environment.

    Attributes:
        player_x: Column of the player.
        player_y: Row of the player.
        shot_timer: Unused; stays at 0.
        spawn_speed: Current number of steps between entity spawns.
        spawn_timer: Steps left until the next entity spawns.
        move_speed: Current number of steps between entity moves.
        move_timer: Steps left until the entities move next.
        ramp_timer: Countdown to the next difficulty increase.
        ramp_index: Number of difficulty increases so far.
        entities: Array of shape `(8, 5)` with one entity slot for each of the rows 1
            to 8. Columns are `[x, y, direction, is_gold, active]`, where `direction`
            is 1 for entities moving right and 0 for entities moving left.
        time: Number of steps taken in the current episode.
        terminal: Whether the episode has terminated.
    """

    player_x: int
    player_y: int
    shot_timer: int
    spawn_speed: int
    spawn_timer: int
    move_speed: int
    move_timer: int
    ramp_timer: int
    ramp_index: int
    entities: jax.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the Asterix environment.

    Attributes:
        ramping: Whether the difficulty increases over the course of an episode.
        ramp_interval: Start value of the countdown between difficulty increases.
        init_spawn_speed: Initial number of steps between entity spawns.
        init_move_interval: Initial number of steps between entity moves.
        shot_cool_down: Unused.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    ramping: bool = True
    ramp_interval: int = 100
    init_spawn_speed: int = 10
    init_move_interval: int = 5
    shot_cool_down: int = 5
    max_steps_in_episode: int = 1000


class MinAsterix(environment.Environment[EnvState, EnvParams]):
    """Collect treasure while dodging enemies that cross the screen.

    JAX port of the MinAtar Asterix environment. The player starts at column 5, row 5
    of a 10x10 grid and moves freely in the four cardinal directions within rows 1 to
    8. Enemies and treasure spawn at the left and right edges of these rows and cross
    the grid horizontally, with at most one entity per row; each spawned entity is
    treasure with probability 1/3. If `params.ramping` is set, the difficulty
    increases periodically: entities spawn and move faster.

    - Observation: `Box` of shape `(10, 10, 4)` with binary channels: 0 player, 1
      enemy, 2 trail (the cell behind each enemy and treasure, indicating its
      direction), 3 treasure.
    - Actions: `Discrete(5)` with the minimal action set: 0 no-op, 1 left, 2 up, 3
      right, 4 down. `Discrete(6)` with the full action set, which adds 5 (fire)
      without any effect.
    - Reward: +1 for each treasure collected.
    - Termination: the player makes contact with an enemy.
    - Truncation: after `params.max_steps_in_episode` steps (1000 by default).

    Args:
        use_minimal_action_set: Whether to expose only the actions that have an
            effect in this game instead of all six MinAtar actions.

    References:
        - [Young & Tian (2019)](https://arxiv.org/abs/1903.03176)
        - [MinAtar implementation](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/asterix.py)
    """

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 4)
        # Full action set: ['n','l','u','r','d','f']
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        # Minimal action set: ['n', 'l', 'u', 'r', 'd']
        self.minimal_action_set = jnp.array([0, 1, 2, 3, 4])
        # Set active action set for environment
        # If minimal map to integer in full action set
        if use_minimal_action_set:
            self.action_set = self.minimal_action_set
        else:
            self.action_set = self.full_action_set

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Asterix-MinAtar."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Spawn enemy if timer up - sample at each step & select based on timer
        spawn_entities_now = state.spawn_timer == 0
        entity, slot = spawn_entity(key, state)
        entities = jax.lax.select(
            spawn_entities_now,
            state.entities.at[slot].set(entity),
            state.entities,
        )
        spawn_timer = jax.lax.select(
            spawn_entities_now, state.spawn_speed, state.spawn_timer
        )
        state = state.replace(entities=entities, spawn_timer=spawn_timer)

        # Update state of the players
        a = self.action_set[action]
        state = step_agent(state, a)

        # Update entities, get reward and figure out termination
        state, reward, done = step_entities(state)

        # Update timers and ramping condition check
        state = step_timers(state, params)

        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1, terminal=done)
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state to the fixed initial configuration."""
        state = EnvState(
            player_x=5,
            player_y=5,
            shot_timer=0,
            spawn_speed=params.init_spawn_speed,
            spawn_timer=params.init_spawn_speed,
            move_speed=params.init_move_interval,
            move_timer=params.init_move_interval,
            ramp_timer=params.ramp_interval,
            ramp_index=0,
            entities=jnp.zeros((8, 5), dtype=int),
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        # Add a 5th channel to help with not used entities
        obs = jnp.zeros((10, 10, 5), dtype=jnp.int32)
        # Set the position of the agent in the grid
        obs = obs.at[state.player_y, state.player_x, 0].set(1)
        # Loop over entity identities and set entity locations
        for i in range(state.entities.shape[0]):
            x = state.entities[i, :]
            # Enemy channel 1, Trail channel 2, Gold channel 3, Not used 4
            c = 3 * x[3] + 1 * (1 - x[3])
            c_eff = c * x[4] + 4 * (1 - x[4])
            obs = obs.at[x[1], x[0], c_eff].set(1)

            back_x = (x[0] - 1) * x[2] + (x[0] + 1) * (1 - x[2])
            leave_trail = jnp.logical_and(back_x >= 0, back_x <= 9)
            c_eff = 2 * x[4] + 4 * (1 - x[4])
            obs = obs.at[x[1], back_x, c_eff].set(leave_trail)
        return obs[:, :, :4].astype(jnp.float32)

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return state.terminal

    @property
    def name(self) -> str:
        """Environment name."""
        return "Asterix-MinAtar"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "player_x": spaces.Discrete(10),
                "player_y": spaces.Discrete(10),
                "shot_timer": spaces.Discrete(1000),
                "spawn_speed": spaces.Discrete(1000),
                "spawn_timer": spaces.Discrete(1000),
                "move_speed": spaces.Discrete(1000),
                "move_timer": spaces.Discrete(1000),
                "ramp_timer": spaces.Discrete(1000),
                "ramp_index": spaces.Discrete(1000),
                "entities": spaces.Box(0, 1, (8, 5)),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(state: EnvState, action: jax.Array) -> EnvState:
    """Update the position of the agent."""
    # Resolve player action via implicit conditional updates of coordinates
    player_x = (
        jnp.maximum(0, state.player_x - 1) * (action == 1)  # l
        + jnp.minimum(9, state.player_x + 1) * (action == 3)  # r
        + state.player_x * jnp.logical_and(action != 1, action != 3)
    )  # others

    player_y = (
        jnp.maximum(1, state.player_y - 1) * (action == 2)  # u
        + jnp.minimum(8, state.player_y + 1) * (action == 4)  # d
        + state.player_y * jnp.logical_and(action != 2, action != 4)
    )  # others
    return state.replace(player_x=player_x, player_y=player_y)


def spawn_entity(key: jax.Array, state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Spawn new enemy or treasure at random location with random direction."""
    key_lr, key_gold, key_slot = jax.random.split(key, 3)
    lr = jax.random.choice(key_lr, jnp.array([1, 0]))
    is_gold = jax.random.choice(
        key_gold, jnp.array([1, 0]), p=jnp.array([1 / 3, 2 / 3])
    )
    x = (1 - lr) * 9  # l-to-r starts at 0
    # Entities are represented as 5 dimensional arrays
    # 0: Position x, 1: Position y (slot + 1), 2: lr (from l to r dir),
    # 3: Gold indicator
    # 4: whether entity is filled/not an open slot

    # Sampling problem: Need to get rid of jnp.where due to concretization
    # Sample random order of entries to go through
    # Check if element is free with while loop and stop if position is found
    # or all elements have been checked
    state_entities = state.entities[:, 4]  # Only use col 4 indicating free
    slot, free = while_sample_slots(key_slot, state_entities)
    entity = jnp.array([x, slot + 1, lr, is_gold, free])
    return entity, slot


def while_sample_slots(
    key: jax.Array, state_entities: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Go through random order of slots until slot is found that is free."""
    init_val = jnp.array([0, 0])
    # Sample random order of slot entries to go through - hack around jnp.where
    order_to_go_through = jax.random.permutation(key, jnp.arange(8))
    perm_entities = state_entities[order_to_go_through]

    def condition_to_check(val):
        # Check if we haven't gone through all possible slots and whether free
        return jnp.logical_and(val[0] < 7, val[1] == 0)

    def update(val):
        # Increase list counter - slot that has been checked
        val = val.at[0].set(val[0] + 1)
        # Check if slot is still free
        free = perm_entities[val[0]] == 0
        val = val.at[1].set(free)
        return val

    id_and_free = jax.lax.while_loop(condition_to_check, update, init_val)
    # Return slot id and whether it is free
    slot_id = order_to_go_through[id_and_free[0]]
    free_slot = id_and_free[1]
    return slot_id, free_slot


def step_entities(
    state: EnvState,
) -> tuple[EnvState, jax.Array, bool]:
    """Update positions of the entities and return reward, done."""
    done, reward = 0, jnp.array(0)
    # Loop over entities and check for collisions - either gold or enemy
    entities = state.entities
    for i in range(8):
        x = entities[i]
        slot_filled = x[4] != 0
        # Get boolean for any collision with either gold or enemy
        coords = jnp.logical_and(x[0] == state.player_x, x[1] == state.player_y)
        collision = jnp.logical_and(coords, slot_filled)
        # If collision with gold: empty gold and give positive reward
        collision_gold = jnp.logical_and(collision, x[3])
        reward += collision_gold
        # Set row i to zeros if collision with gold
        entities = entities.at[i].set(x * (1 - collision_gold))

        # If collision with enemy: terminate the episode
        collision_enemy = jnp.logical_and(collision, 1 - x[3])
        done += collision_enemy

    # Loop over entities and move them in direction
    time_to_move = state.move_timer == 0
    move_timer = jax.lax.select(time_to_move, state.move_speed, state.move_timer)

    old_entities = entities
    for i in range(8):
        x = entities[i]
        slot_filled = x[4] != 0
        lr = x[2]
        # Update position left and right move
        x = x.at[0].set(jax.lax.select(slot_filled, x[0] + 1 * lr - 1 * (1 - lr), x[0]))

        # Update if entity moves out of the frame - reset everything to zeros
        outside_of_frame = jnp.logical_or(x[0] < 0, x[0] > 9)
        entities = jax.lax.select(
            time_to_move,
            entities.at[i].set(x * slot_filled * (1 - outside_of_frame)),
            old_entities,
        )

        # Update if entity moves into the player after its state is updated
        coords = jnp.logical_and(x[0] == state.player_x, x[1] == state.player_y)
        collision = jnp.logical_and(coords, slot_filled)
        # If collision with gold: empty gold and give positive reward
        collision_gold = jnp.logical_and(collision, x[3])
        reward += jax.lax.select(time_to_move, collision_gold, False) * 1
        entities = jax.lax.select(
            time_to_move,
            entities.at[i].set(entities[i] * (1 - collision_gold)),
            old_entities,
        )
        # If collision with enemy: terminate the episode
        collision_enemy = jnp.logical_and(collision, 1 - x[3])
        done += jax.lax.select(time_to_move, collision_enemy, False)
    return (
        state.replace(entities=entities, move_timer=move_timer),
        reward,
        jnp.bool_(done > 0),
    )


def step_timers(state: EnvState, params: EnvParams) -> EnvState:
    """Update various timers and check the ramping condition."""
    spawn_timer = state.spawn_timer - 1
    move_timer = state.move_timer - 1

    # Ramp difficulty if interval has elapsed
    ramp_cond = jnp.logical_and(
        params.ramping,
        jnp.logical_or(state.spawn_speed > 1, state.move_speed > 1),
    )
    # 1. Update ramp_timer
    timer_cond = jnp.logical_and(ramp_cond, state.ramp_timer >= 0)
    ramp_timer = jax.lax.select(timer_cond, state.ramp_timer - 1, params.ramp_interval)
    # 2. Update move_speed
    move_speed_cond = jnp.logical_and(
        jnp.logical_and(ramp_cond, 1 - timer_cond),
        jnp.logical_and(state.move_speed, state.ramp_index % 2),
    )
    move_speed = state.move_speed - move_speed_cond
    # 3. Update spawn_speed
    spawn_speed_cond = jnp.logical_and(
        jnp.logical_and(ramp_cond, 1 - timer_cond), state.spawn_speed > 1
    )
    spawn_speed = state.spawn_speed - spawn_speed_cond
    # 4. Update ramp_index
    ramp_index = state.ramp_index + jnp.logical_and(ramp_cond, 1 - timer_cond)
    return state.replace(
        spawn_timer=spawn_timer,
        move_timer=move_timer,
        ramp_timer=ramp_timer,
        move_speed=move_speed,
        spawn_speed=spawn_speed,
        ramp_index=ramp_index,
    )

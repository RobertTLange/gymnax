import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class MinSpaceInvaders(environment.Environment):
    """
    JAX Compatible version of Freeway MinAtar environment. Source:
    github.com/kenjyoung/MinAtar/blob/master/minatar/environments/space_invaders.py

    ENVIRONMENT DESCRIPTION - 'SpaceInvaders-MinAtar'
    - Player controls cannon at bottom of screen and can shoot bullets at aliens
    - Aliens move across screen until one of them hits the edge.
    - At this point all move down and switch directions.
    - Current alien dir indicated by 2 channels (left/right) - active at position.
    - Reward of +1 is given each time alien is shot and alien is removed.
    - Aliens will also shoot bullets back at player.
    - Alien speed increases when only few of them are left.
    - When only one alien is left, it will move at one cell per frame.
    - When wave of aliens is cleared, slightly faster new one will spawn.
    - Termination occurs when an alien or bullet hits the player.
    - Channels are encoded as follows: 'cannon':0, 'alien':1, 'alien_left':2,
    - 'alien_right':3, 'friendly_bullet':4, 'enemy_bullet':5
    - Observation has dimensionality (10, 10, 6)
    - Actions are encoded as follows: ['n','l','r','f']
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (10, 10, 6)

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "shot_cool_down": 5,
            "enemy_move_interval": 12,
            "enemy_shot_interval": 10,
            "max_steps_in_episode": 100,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        # Resolve player action - fire, left, right.
        state = step_agent(action, state, params)
        # Update aliens - border and collision check.
        state = step_aliens(state)
        # Update aliens - shooting check and calculate rewards.
        state, reward = step_shoot(state, params)

        # Update various timers & evaluate all terminal conditions
        state["shot_timer"] = state["shot_timer"] - (state["shot_timer"] > 0)
        state["alien_move_timer"] -= 1
        state["alien_shot_timer"] -= 1

        # Reset alien map and increase speed if map is cleared
        reset_map_cond = jnp.count_nonzero(state["alien_map"]) == 0
        ramping_cond = jnp.logical_and(
            state["enemy_move_interval"] > 6, state["ramping"]
        )
        reset_ramp_cond = jnp.logical_and(reset_map_cond, ramping_cond)
        state["enemy_move_interval"] = state["enemy_move_interval"] - reset_ramp_cond
        state["ramp_index"] = state["ramp_index"] + reset_ramp_cond
        state["alien_map"] = (
            reset_map_cond
            * jax.ops.index_update(state["alien_map"], jax.ops.index[0:4, 2:8], 1)
            + (1 - reset_map_cond) * state["alien_map"]
        )

        # Check game condition & no. steps for termination condition
        state["time"] += 1
        done = self.is_terminal(state, params)
        state["terminal"] = done
        info = {"discount": 1 - done}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        alien_map = jnp.zeros((10, 10))
        alien_map = jax.ops.index_update(alien_map, jax.ops.index[0:4, 2:9], 1)
        state = {
            "pos": 5,
            "f_bullet_map": jnp.zeros((10, 10)),
            "e_bullet_map": jnp.zeros((10, 10)),
            "alien_map": alien_map,
            "alien_dir": -1,
            "enemy_move_interval": params["enemy_move_interval"],
            "alien_move_timer": params["enemy_move_interval"],
            "alien_shot_timer": params["enemy_shot_interval"],
            "ramp_index": 0,
            "shot_timer": 0,
            "ramping": True,
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((10, 10, 6), dtype=bool)
        # Update cannon, aliens - left + right dir, friendly + enemy bullet
        obs = jax.ops.index_update(obs, jax.ops.index[9, state["pos"], 0], 1)
        obs = jax.ops.index_update(obs, jax.ops.index[:, :, 1], state["alien_map"])
        left_dir_cond = state["alien_dir"] < 0
        obs = left_dir_cond * jax.ops.index_update(
            obs, jax.ops.index[:, :, 2], state["alien_map"]
        ) + (1 - left_dir_cond) * jax.ops.index_update(
            obs, jax.ops.index[:, :, 3], state["alien_map"]
        )
        obs = jax.ops.index_update(obs, jax.ops.index[:, :, 4], state["f_bullet_map"])
        obs = jax.ops.index_update(obs, jax.ops.index[:, :, 5], state["e_bullet_map"])
        return obs

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done_steps = state["time"] > params["max_steps_in_episode"]
        return jnp.logical_or(done_steps, state["terminal"])

    @property
    def name(self) -> str:
        """Environment name."""
        return "SpaceInvaders-MinAtar"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Discrete(10),
                "f_bullet_map": spaces.Box(0, 1, (10, 10)),
                "e_bullet_map": spaces.Box(0, 1, (10, 10)),
                "alien_map": spaces.Box(0, 1, (10, 10)),
                "alien_dir": spaces.Box(-1, 3, ()),
                "enemy_move_interval": spaces.Discrete(params["enemy_move_interval"]),
                "alien_move_timer": spaces.Discrete(params["enemy_move_interval"]),
                "alien_shot_timer": spaces.Discrete(params["enemy_shot_interval"]),
                "ramp_index": spaces.Discrete(2),
                "shot_timer": spaces.Discrete(1000),
                "ramping": spaces.Discrete(2),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(action, state, params):
    """Resolve player action - fire, left, right."""
    fire_cond = jnp.logical_and(action == 3, state["shot_timer"] == 0)
    left_cond, right_cond = (action == 1), (action == 2)
    state["f_bullet_map"] = (
        fire_cond
        * jax.ops.index_update(state["f_bullet_map"], jax.ops.index[9, state["pos"]], 1)
        + (1 - fire_cond) * state["f_bullet_map"]
    )
    state["shot_timer"] = (
        fire_cond * params["shot_cool_down"] + (1 - fire_cond) * state["shot_timer"]
    )

    state["pos"] = (
        left_cond * jnp.maximum(0, state["pos"] - 1) + (1 - left_cond) * state["pos"]
    )
    state["pos"] = (
        right_cond * jnp.minimum(9, state["pos"] + 1) + (1 - right_cond) * state["pos"]
    )

    # Update Friendly Bullets and Enemy Bullets
    state["f_bullet_map"] = jnp.roll(state["f_bullet_map"], -1, axis=0)
    state["f_bullet_map"] = jax.ops.index_update(
        state["f_bullet_map"], jax.ops.index[9, :], 0
    )
    state["e_bullet_map"] = jnp.roll(state["e_bullet_map"], 1, axis=0)
    state["e_bullet_map"] = jax.ops.index_update(
        state["e_bullet_map"], jax.ops.index[0, :], 0
    )
    bullet_terminal = state["e_bullet_map"][9, state["pos"]]
    state["terminal"] = jnp.logical_or(state["terminal"], bullet_terminal)
    return state


def step_aliens(state):
    """Update aliens - border and collision check."""
    alien_terminal_1 = state["alien_map"][9, state["pos"]]
    alien_move_cond = state["alien_move_timer"] == 0

    state["alien_move_timer"] = (
        alien_move_cond
        * jnp.minimum(
            jnp.count_nonzero(state["alien_map"]), state["enemy_move_interval"]
        )
        + (1 - alien_move_cond) * state["alien_move_timer"]
    )

    cond1 = jnp.logical_and(
        jnp.sum(state["alien_map"][:, 0]) > 0, state["alien_dir"] < 0
    )
    cond2 = jnp.logical_and(
        jnp.sum(state["alien_map"][:, 9]) > 0, state["alien_dir"] > 0
    )
    alien_border_cond = jnp.logical_and(alien_move_cond, jnp.logical_or(cond1, cond2))
    state["alien_dir"] = (
        alien_border_cond * -1 * state["alien_dir"]
        + (1 - alien_border_cond) * state["alien_dir"]
    )
    alien_terminal_2 = jnp.logical_and(
        alien_border_cond, jnp.sum(state["alien_map"][9, :]) > 0
    )

    state["alien_map"] = (
        alien_move_cond
        * (
            alien_border_cond * jnp.roll(state["alien_map"], 1, axis=0)
            + (1 - alien_border_cond)
            * jnp.roll(state["alien_map"], state["alien_dir"], axis=1)
        )
        + (1 - alien_move_cond) * state["alien_map"]
    )
    alien_terminal_3 = jnp.logical_and(
        alien_move_cond, state["alien_map"][9, state["pos"]]
    )

    # Jointly evaluate the 3 alien terminal conditions
    alien_terminal = (alien_terminal_1 + alien_terminal_2 + alien_terminal_3) > 0
    state["terminal"] = jnp.logical_or(state["terminal"], alien_terminal)
    return state


def step_shoot(state, params):
    """Update aliens - shooting check and calculate rewards."""
    reward = 0
    alien_shot_cond = state["alien_shot_timer"] == 0
    state["alien_shot_timer"] = (
        alien_shot_cond * params["enemy_shot_interval"]
        + (1 - alien_shot_cond) * state["alien_shot_timer"]
    )
    # nearest_alien has 3 outputs used to update map: [alien_exists, loc, id]
    alien_exists, loc, id = get_nearest_alien(state["pos"], state["alien_map"])
    update_aliens_cond = jnp.logical_and(alien_shot_cond, alien_exists)
    state["e_bullet_map"] = (
        update_aliens_cond
        * jax.ops.index_update(state["e_bullet_map"], jax.ops.index[loc, id], 1)
        + (1 - update_aliens_cond) * state["e_bullet_map"]
    )

    kill_locations = jnp.logical_and(
        state["alien_map"], state["alien_map"] == state["f_bullet_map"]
    )

    # Compute reward based on killed aliens
    reward += jnp.sum(kill_locations)
    # Delete aliens/bullets based on kill_locations elementwise multiplication
    state["alien_map"] = state["alien_map"] * (1 - kill_locations)
    state["f_bullet_map"] = state["f_bullet_map"] * (1 - kill_locations)
    return state, reward


def get_nearest_alien(pos, alien_map):
    """Find alien closest to player in manhattan distance -> shot target."""
    ids = jnp.array([jnp.abs(jnp.array([i for i in range(10)]) - pos)])
    search_order = jnp.argsort(ids).squeeze()
    results_temp = jnp.zeros(3)
    aliens_exist = jnp.sum(alien_map, axis=0) > 0

    # Work around for np.where via element-wise multiplication with ids
    # The output has 3 dims: [alien_exists, location, id]
    counter = 0
    for i in search_order[::-1]:
        locations = alien_map[:, i] * jnp.arange(alien_map[:, i].shape[0])
        aliens_loc = jnp.max(locations)
        results_temp = (
            aliens_exist[i]
            * jax.ops.index_update(
                results_temp,
                jax.ops.index[:],
                jnp.array([aliens_exist[i], aliens_loc, i]),
            )
            + (1 - aliens_exist[i]) * results_temp
        )
        counter += 1
    results_temp = jnp.array(results_temp, dtype=int)
    # Loop over results in reverse order
    return results_temp[0], results_temp[1], results_temp[2]

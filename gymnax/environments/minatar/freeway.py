import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class MinFreeway(environment.Environment):
    """
    JAX Compatible version of Freeway MinAtar environment. Source:
    github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py

    ENVIRONMENT DESCRIPTION - 'Freeway-MinAtar'
    - Player starts at bottom of screen and can travel up/down.
    - Player speed is restricted s.t. player only moves every 3 frames.
    - Reward +1 given when player reaches top of screen -> returns to bottom.
    - 8 cars travel horizontally on screen and teleport to other side at edge.
    - When player is hit by a car, he is returned to the bottom of the screen.
    - Car direction and speed are indicated by 5 trail channels.
    - Each time player reaches top of screen, car speeds are randomized.
    - Termination occurs after 2500 frames.
    - Channels are encoded as follows: 'chicken':0, 'car':1, 'speed1':2,
    - 'speed2':3, 'speed3':4, 'speed4':5, 'speed5':6
    - Observation has dimensionality (10, 10, 4)
    - Actions are encoded as follows: ['n', 'u', 'd']
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (10, 10, 7)

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "player_speed": 3,
            "time_limit": 2500,
            "max_steps_in_episode": 100,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        # 1. Update position of agent only if timer condition is met!
        state, reward, win_cond = step_agent(action, state, params)

        # 2. Sample a new configuration for the cars if agent 'won'
        # Note: At each step we are sampling speed and dir to avoid if cond
        # by masking - still faster after compilation than numpy version!
        key_speed, key_dirs = jax.random.split(key)
        speeds = jax.random.randint(key_speed, shape=(8,), minval=1, maxval=6)
        directions = jax.random.choice(key_dirs, jnp.array([-1, 1]), shape=(8,))
        win_cars = randomize_cars(speeds, directions, state["cars"], False)
        state["cars"] = jax.ops.index_update(
            state["cars"],
            jax.ops.index[:, :],
            win_cars * win_cond + state["cars"] * (1 - win_cond),
        )

        # 3. Update cars and check for collisions! - respawn agent at bottom
        state = step_cars(state)

        # Check game condition & no. steps for termination condition
        state["time"] += 1
        done = self.is_terminal(state, params)
        state["terminal"] = done
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        # Sample the initial speeds and directions of the cars
        key_speed, key_dirs = jax.random.split(key)
        speeds = jax.random.randint(key_speed, shape=(8,), minval=1, maxval=6)
        directions = jax.random.choice(key_dirs, jnp.array([-1, 1]), shape=(8,))

        state = {
            "pos": 9,
            "cars": randomize_cars(
                speeds, directions, jnp.zeros((8, 4), dtype=int), True
            ),
            "move_timer": params["player_speed"],
            "terminate_timer": params["time_limit"],
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(self.obs_shape, dtype=bool)
        # Set the position of the chicken agent, cars, and trails
        obs = jax.ops.index_update(obs, jax.ops.index[state["pos"], 4, 0], 1)
        for car_id in range(8):
            car = state["cars"][car_id]
            obs = jax.ops.index_update(obs, jax.ops.index[car[1], car[0], 1], 1)
            # Boundary conditions for cars
            back_x = (car[3] > 0) * (car[0] - 1) + (1 - (car[3] > 0)) * (car[0] + 1)
            left_out = back_x < 0
            right_out = back_x > 9
            back_x = left_out * 9 + (1 - left_out) * back_x
            back_x = right_out * 0 + (1 - right_out) * back_x
            # Set trail to be on
            trail_channel = (
                2 * (jnp.abs(car[3]) == 1)
                + 3 * (jnp.abs(car[3]) == 2)
                + 4 * (jnp.abs(car[3]) == 3)
                + 5 * (jnp.abs(car[3]) == 4)
                + 6 * (jnp.abs(car[3]) == 5)
            )
            obs = jax.ops.index_update(
                obs, jax.ops.index[car[1], back_x, trail_channel], 1
            )
        return obs

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        done_steps = state["time"] > params["max_steps_in_episode"]
        done_timer = state["terminate_timer"] < 0
        return jnp.logical_or(done_steps, done_timer)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Freeway-MinAtar"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Discrete(10),
                "cars": spaces.Box(0, 1, jnp.zeros((8, 4)), dtype=jnp.int_),
                "move_timer": spaces.Discrete(params["player_speed"]),
                "terminate_timer": spaces.Discrete(params["time_limit"]),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(action: int, state: dict, params: dict):
    """Perform 1st part of step transition for agent."""
    cond_up = jnp.logical_and(action == 1, state["move_timer"] == 0)
    cond_down = jnp.logical_and(action == 2, state["move_timer"] == 0)
    any_cond = jnp.logical_or(cond_up, cond_down)
    state_up = jnp.maximum(0, state["pos"] - 1)
    state_down = jnp.minimum(9, state["pos"] + 1)
    state["pos"] = (
        (1 - any_cond) * state["pos"] + cond_up * state_up + cond_down * state_down
    )
    state["move_timer"] = (1 - any_cond) * state["move_timer"] + any_cond * params[
        "player_speed"
    ]
    # Check win cond. - increase reward, randomize cars, reset agent position
    win_cond = state["pos"] == 0
    reward = win_cond
    state["pos"] = 9 * win_cond + state["pos"] * (1 - win_cond)
    return state, reward, win_cond


def step_cars(state: dict):
    """Perform 3rd part of step transition for car."""
    # Update cars and check for collisions! - respawn agent at bottom
    for car_id in range(8):
        # Check for agent collision with car and if so reset agent
        collision_cond = jnp.logical_and(
            state["cars"][car_id][0] == 4, state["cars"][car_id][1] == state["pos"]
        )
        state["pos"] = 9 * collision_cond + state["pos"] * (1 - collision_cond)

        # Check for exiting frame, reset car and then check collision again
        car_cond = state["cars"][car_id][2] == 0
        upd_2 = (
            car_cond * jnp.abs(state["cars"][car_id][3])
            + (1 - car_cond) * state["cars"][car_id][2]
        )
        state["cars"] = jax.ops.index_update(
            state["cars"], jax.ops.index[car_id, 2], upd_2
        )
        upd_0 = (
            car_cond
            * (
                state["cars"][car_id][0]
                + 1 * (state["cars"][car_id][3] > 0)
                - 1 * (1 - (state["cars"][car_id][3] > 0))
            )
            + (1 - car_cond) * state["cars"][car_id][0]
        )
        state["cars"] = jax.ops.index_update(
            state["cars"], jax.ops.index[car_id, 0], upd_0
        )

        cond_sm_0 = jnp.logical_and(car_cond, state["cars"][car_id][0] < 0)
        upd_0_sm = cond_sm_0 * 9 + (1 - cond_sm_0) * state["cars"][car_id][0]
        state["cars"] = jax.ops.index_update(
            state["cars"], jax.ops.index[car_id, 0], upd_0_sm
        )
        cond_gr_9 = jnp.logical_and(car_cond, state["cars"][car_id][0] > 9)
        upd_0_gr = cond_gr_9 * 0 + (1 - cond_gr_9) * state["cars"][car_id][0]
        state["cars"] = jax.ops.index_update(
            state["cars"], jax.ops.index[car_id, 0], upd_0_gr
        )

        # Check collision after car position update - respawn agent
        # Note: Need to reevaluate collision condition since cars change!
        collision_cond = jnp.logical_and(
            state["cars"][car_id][0] == 4, state["cars"][car_id][1] == state["pos"]
        )
        cond_pos = jnp.logical_and(car_cond, collision_cond)
        state["pos"] = cond_pos * 9 + (1 - cond_pos) * state["pos"]
        # Move car if no previous car_cond update
        alt_upd_2 = car_cond * state["cars"][car_id][2] + (1 - car_cond) * (
            state["cars"][car_id][2] - 1
        )
        state["cars"] = jax.ops.index_update(
            state["cars"], jax.ops.index[car_id, 2], alt_upd_2
        )
    # 4. Update various timers
    state["move_timer"] -= state["move_timer"] > 0
    state["terminate_timer"] -= 1
    return state


def randomize_cars(speeds, directions, old_cars, initialize=0):
    """Randomize car speeds & directions. Reset position if initialize."""
    speeds_new = directions * speeds
    new_cars = jnp.zeros((8, 4), dtype=int)

    # Loop over all 8 cars and set their data
    for i in range(8):
        # Reset both speeds, directions and positions
        new_cars = jax.ops.index_update(
            new_cars,
            jax.ops.index[i, :],
            [0, i + 1, jnp.abs(speeds_new[i]), speeds_new[i]],
        )
        # Reset only speeds and directions
        old_cars = jax.ops.index_update(
            old_cars, jax.ops.index[i, 2:4], [jnp.abs(speeds_new[i]), speeds_new[i]]
        )

    # Mask the car array manipulation according to initialize
    cars = initialize * new_cars + (1 - initialize) * old_cars
    return jnp.array(cars, dtype=jnp.int_)

import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey


class MinSeaquest(environment.Environment):
    """
    JAX Compatible version of Seaquest MinAtar environment. Source:
    github.com/kenjyoung/MinAtar/blob/master/minatar/environments/seaquest.py

    ENVIRONMENT DESCRIPTION - 'Seaquest-MinAtar'
    - Player controls submarine consisting of two cells - front and back.
    - Player can fire bullets from front of submarine.
    - Enemies consist of submarines [shoot] and fish [don't shoot].
    - A reward of +1 is given whenever enemy is struck by bullet and removed.
    - Player can pick up drivers which increments a bar indicated by a channel.
    - Player has limited oxygen supply indicated by bar in separate channel.
    - Oxygen degrades over time. Can be restored:
    - If player moves to top of screen and has at least 1 rescued driver on board.
    - When surfacing with less than 6, one diver is removed.
    - When surfacing w. 6, remove all divers. R for each active cell in oxygen bar.
    - Each time the player surfaces increase difficulty by increasing
      the spawn rate and movement speed of enemies.
    - Termination occurs when player is hit by an enemy fish, sub or bullet
    - Or when oxygen reached 0.
    - Or when the layer attempts to surface with no rescued divers.
    - Enemy and diver directions are indicated by a trail channel active
    - in their previous location to reduce partial observability.

    - Channels are encoded as follows: 'sub_front':0, 'sub_back':1,
                                       'friendly_bullet':2, 'trail':3,
                                       'enemy_bullet':4, 'enemy_fish':5,
                                       'enemy_sub':6, 'oxygen_guage':7,
                                       'diver_guage':8, 'diver':9
    - Observation has dimensionality (10, 10, 10)
    - Actions are encoded as follows: ['n','l','u','r','d','f']
    """
    def __init__(self):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"ramp_interval": 100,
                                      "max_oxygen": 200,
                                      "init_spawn_speed": 20,
                                      "diver_spawn_speed": 30,
                                      "init_move_interval": 5,
                                      "shot_cool_down": 5,
                                      "enemy_shot_interval": 10,
                                      "enemy_move_interval": 5,
                                      "diver_move_interval": 5,
                                      "obs_shape": (10, 10, 10),
                                      "max_steps_in_episode": 100})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        reward = 0
        # Check game condition & no. steps for termination condition
        state["time"] += 1
        done = self.is_terminal(state)
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done, info)

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        state = {"oxygen": self.env_params["max_oxygen"],
                 "diver_count": 0,
                 "sub_x": 5,
                 "sub_y": 0,
                 "sub_or": 0,
                 "f_bullets": jnp.zeros((10, 10)),
                 "e_bullets": jnp.zeros((10, 10)),
                 "e_fish": jnp.zeros((10, 10)),
                 "e_subs": jnp.zeros((10, 10)),
                 "divers": jnp.zeros((10, 10)),
                 "e_spawn_speed": self.env_params["init_spawn_speed"],
                 "e_spawn_timer": self.env_params["init_spawn_speed"],
                 "d_spawn_timer": self.env_params["diver_spawn_speed"],
                 "move_speed": self.env_params["init_move_interval"],
                 "ramp_index": 0,
                 "shot_timer": 0,
                 "surface": 1,
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        obs = jnp.zeros((10, 10, 10), dtype=bool)
        # Set agents sub-front and back, oxygen_gauge and diver_gauge
        obs = jax.ops.index_update(obs, jax.ops.index[state['sub_y'],
                                                      state['sub_x'], 0], 1)
        back_x = ((state["sub_x"] - 1) * state["sub_or"] +
                  (state["sub_x"] + 1) * (1 - state["sub_or"]))
        obs = jax.ops.index_update(obs, jax.ops.index[state['sub_y'],
                                                      back_x, 1], 1)
        obs = jax.ops.index_update(obs, jax.ops.index[9,
                                0:state["oxygen"]*10//
                                self.env_params["max_oxygen"], 7], 1)
        obs = jax.ops.index_update(obs, jax.ops.index[9, 9-
                                                      state["diver_count"]:9,
                                                      8], 1)

        # Set friendly bulltes, enemy bullets, enemy fish+trail, enemey sub+trail
        for bullet in state["f_bullets"]:
            obs = jax.ops.index_update(obs, jax.ops.index[bullet[1], bullet[0],
                                                          2], 1)
        for bullet in state["e_bullets"]:
            obs = jax.ops.index_update(obs, jax.ops.index[bullet[1], bullet[0],
                                                          4], 1)

        for fish in state["e_fish"]:
            obs = jax.ops.index_update(obs, jax.ops.index[fish[1], fish[0], 5], 1)
            back_x = ((fish[0] - 1) * fish[2] +
                      (fish[0] + 1) * (1 - fish[2]))
            border_cond = jnp.logical_and(back_x >= 0, back_x <= 9)
            obs = (border_cond * jax.ops.index_update(obs, jax.ops.index[fish[1],
                   back_x, 3], 1) + (1 - border_cond) * obs)

        for sub in state["e_subs"]:
            obs = jax.ops.index_update(obs, jax.ops.index[sub[1], sub[0], 6], 1)
            back_x = ((sub[0] - 1) * sub[2] +
                      (sub[0] + 1) * (1 - sub[2]))
            border_cond = jnp.logical_and(back_x >= 0, back_x <= 9)
            obs = (border_cond * jax.ops.index_update(obs, jax.ops.index[sub[1],
                   back_x, 3], 1) + (1 - border_cond) * obs)

        for diver in state["divers"]:
            obs = jax.ops.index_update(obs, jax.ops.index[diver[1],diver[0], 9], 1)
            back_x = ((diver[0] - 1) * diver[2] +
                      (diver[0] + 1) * (1 - diver[2]))
            border_cond = jnp.logical_and(back_x >= 0, back_x <= 9)
            obs = (border_cond * jax.ops.index_update(obs, jax.ops.index[diver[1],
                   back_x, 3], 1) + (1 - border_cond) * obs)
        return obs

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        return done_steps

    @property
    def name(self) -> str:
        """ Environment name. """
        return "Seaquest-MinAtar"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(6)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(0, 1, self.env_params["obs_shape"])

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"oxygen": spaces.Discrete(self.env_params["max_oxygen"]),
             "diver_count": spaces.Discrete(20),
             "sub_x": spaces.Discrete(10),
             "sub_y": spaces.Discrete(10),
             "sub_or": spaces.Discrete(2),
             "f_bullets": spaces.Box(0, 1, (10, 10)),
             "e_bullets": spaces.Box(0, 1, (10, 10)),
             "e_fish": spaces.Box(0, 1, (10, 10)),
             "e_subs": spaces.Box(0, 1, (10, 10)),
             "divers": spaces.Box(0, 1, (10, 10)),
             "e_spawn_speed": spaces.Discrete(self.env_params["init_spawn_speed"]),
             "e_spawn_timer": spaces.Discrete(self.env_params["init_spawn_speed"]),
             "d_spawn_timer": spaces.Discrete(self.env_params["diver_spawn_speed"]),
             "move_speed":  spaces.Discrete(1000),
             "ramp_index":  spaces.Discrete(1000),
             "shot_timer": spaces.Discrete(self.env_params["shot_cool_down"]),
             "surface": spaces.Discrete(2),
             "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})


def step_agent():
    return

def step_bullets():
    return

def step_divers():
    return

def step_e_subs():
    return

def step_timers():
    return

def spawn_enemy():
    return

def spawn_diver():
    return

def surface():
    return

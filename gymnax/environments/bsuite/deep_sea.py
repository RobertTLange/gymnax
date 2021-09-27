import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey



class DeepSea(environment.Environment):
    """
    JAX Compatible version of DeepSea bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py
    """
    def __init__(self):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"size": 8,
                                      "deterministic": True,
                                      "sample_action_map": False,
                                      "unscaled_move_cost": 0.01,
                                      "randomize_actions": False,
                                      "action_mapping": jnp.ones([8, 8]),
                                      "max_steps_in_episode": 2000})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        # Pull out randomness for easier testing
        rng_reward, rng_trans = jax.random.split(key)
        rand_reward = jax.random.normal(rng_reward, shape=(1,))
        rand_trans_cond = (jax.random.uniform(rng_trans, shape=(1,), minval=0,
                                              maxval=1)
                           > 1/self.env_params["size"])

        action_right = (action ==
                        state["action_mapping"][state["row"], state["column"]])
        right_rand_cond = jnp.logical_or(rand_trans_cond,
                                         self.env_params["deterministic"])
        right_cond = jnp.logical_and(action_right, right_rand_cond)

        reward, denoised_return = step_reward(state, action_right, right_cond,
                                              rand_reward, self.env_params)
        column, row, bad_episode = step_transition(state, action_right,
                                                   right_cond, self.env_params)
        state = {"row": row,
                 "column": column,
                 "bad_episode": bad_episode,
                 "total_bad_episodes": state["total_bad_episodes"],
                 "denoised_return": denoised_return,
                 "optimal_return": state["optimal_return"],
                 "action_mapping": state["action_mapping"],
                 "time": state["time"] + 1}

        # Check row condition & no. steps for termination condition
        done = self.is_terminal(state)
        state["total_bad_episodes"] += done * state["bad_episode"]
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done, info)

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        optimal_no_cost = ((1 - self.env_params["deterministic"])
                            * (1 - 1 / self.env_params["size"])
                            ** (self.env_params["size"] - 1)
                            + self.env_params["deterministic"] * 1.)
        optimal_return = optimal_no_cost - self.env_params["unscaled_move_cost"]

        a_map_rand = jax.random.bernoulli(key, 0.5,
                                          (self.env_params["size"],
                                           self.env_params["size"]))
        a_map_determ = jnp.ones([self.env_params["size"],
                                 self.env_params["size"]])

        new_a_map_cond = jnp.logical_and(1-self.env_params["deterministic"],
                                         self.env_params["sample_action_map"])
        old_a_map_cond = jnp.logical_and(1-self.env_params["deterministic"],
                                         1-self.env_params["sample_action_map"])
        action_mapping = (self.env_params["deterministic"] * a_map_determ
                          + new_a_map_cond * a_map_rand
                          + old_a_map_cond * self.env_params["action_mapping"])

        state = {"row": 0,
                 "column": 0,
                 "bad_episode": False,
                 "total_bad_episodes": 0,
                 "denoised_return": 0,
                 "optimal_return": optimal_return,
                 "action_mapping": action_mapping,
                 "terminal": False,
                 "time": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        obs_end = jnp.zeros(shape=(self.env_params["size"],
                                   self.env_params["size"]),
                            dtype=jnp.float32)
        end_cond = (state["row"] >= self.env_params["size"])
        obs_upd = jax.ops.index_update(obs_end, jax.ops.index[state["row"],
                                                              state["column"]],
                                       1.)
        return end_cond * obs_end + (1 - end_cond) * obs_upd

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        done_row = (state["row"] == self.env_params["size"])
        done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        done = jnp.logical_or(done_row, done_steps)
        return done

    @property
    def name(self) -> str:
        """ Environment name. """
        return "DeepSea-bsuite"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(2)

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(0, 1, (self.env_params["size"],
                                 self.env_params["size"]), jnp.float32)

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"row": spaces.Discrete(self.env_params["size"]),
             "column": spaces.Discrete(self.env_params["size"]),
             "bad_episode": spaces.Discrete(2),
             "total_bad_episodes": spaces.Discrete(2000),
             "denoised_return": spaces.Box(0, 1000, ()),
             "optimal_return": spaces.Box(0, 1000, ()),
             "action_mapping": spaces.Box(0, 1, (self.env_params["size"],
                                                 self.env_params["size"]),
                                          dtype=jnp.int_),
             "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})


def step_reward(state, action_right, right_cond, rand_reward, params):
    """ Get the reward for the selected action. """
    reward = 0.
    # Reward calculation.
    rew_cond = jnp.logical_and(state["column"] == params["size"] - 1,
                               action_right)
    reward += rew_cond
    denoised_return = state["denoised_return"] + rew_cond

    # Noisy rewards on the 'end' of chain.
    col_at_edge = jnp.logical_or(state["column"] == 0,
                                 state["column"] == params["size"] - 1)
    chain_end = jnp.logical_and(state["row"] == params["size"] - 1,
                                col_at_edge)
    det_chain_end = jnp.logical_and(chain_end, params["deterministic"])
    reward += rand_reward * det_chain_end * (1 - params["deterministic"])
    reward -= right_cond * params["unscaled_move_cost"] / params["size"]
    return reward, denoised_return


def step_transition(state, action_right, right_cond, params):
    """ Get the state transition for the selected action. """
    # Standard right path transition
    column = ((1 - right_cond) * state["column"] + right_cond *
              jnp.clip(state["column"] + 1, 0, params["size"] - 1))

    # You were on the right path and went wrong
    right_wrong_cond = jnp.logical_and(1 - action_right,
                                       state["row"] == column)
    bad_episode = ((1 - right_wrong_cond) * state["bad_episode"]
                    + right_wrong_cond * 1)
    column = ((1 - action_right)
               * jnp.clip(state["column"] - 1, 0, params["size"] - 1)
               + action_right * state["column"])
    row = state["row"] + 1
    return column, row, bad_episode

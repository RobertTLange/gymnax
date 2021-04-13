import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict


# JAX Compatible version of DeepSea bsuite environment. Source:
# github.com/deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py

# Default environment parameters
params_deep_sea = FrozenDict({"size": 8,
                              "deterministic": True,
                              "unscaled_move_cost": 0.01,
                              "randomize_actions": True})

#action_mapping = self._mapping_rng.binomial(1, 0.5, [size, size])
action_mapping = jnp.ones([params_deep_sea["size"],
                           params_deep_sea["size"]])


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    reward = 0.
    done = False

    action_right = (action == action_mapping[state["row"], state["column"]])

    # Reward calculation.
    rew_cond = jnp.logical_and(state["column"] == params["size"] - 1,
                               action_right)
    reward += rew_cond
    state["denoised_return"] += rew_cond

    rng_reward, rng_trans = jax.random.split(rng_input)
    # Noisy rewards on the 'end' of chain.
    col_at_edge = jnp.logical_or(state["column"] == 0,
                                 state["column"] == params["size"] - 1)
    chain_end = jnp.logical_and(state["row"] == params["size"] - 1,
                                col_at_edge)
    det_chain_end = jnp.logical_and(chain_end, params["deterministic"])
    reward += jax.random.normal(rng_reward, shape=(1,)) * det_chain_end

    # Transition dynamics
    right_rand_cond = jnp.logical_or(jax.random.uniform(rng_trans, shape=(1,),
                                                        minval=0, maxval=1)
                                > 1/params["size"], params["deterministic"])
    right_cond = jnp.logical_and(action_right, right_rand_cond)

    # Standard right path transition
    state["column"] = ((1 - right_cond) * state["column"] +
                       right_cond *
                       jnp.clip(state["column"] + 1, 0, params["size"] - 1))
    reward -= right_cond * params["unscaled_move_cost"] / params["size"]

    # You were on the right path and went wrong
    right_wrong_cond = jnp.logical_and(1 - action_right,
                                       state["row"] == state["column"])
    state["bad_episode"] = ((1 - right_wrong_cond) * state["bad_episode"]
                            + right_wrong_cond * 1)
    state["column"] = ((1 - action_right)
                       * jnp.clip(state["column"] - 1, 0, params["size"] - 1)
                       + action_right * state["column"])
    state["row"] = state["row"] + 1

    done = (state["row"] == params["size"])
    state["total_bad_episodes"] += done * state["bad_episode"]
    state["terminal"] = done
    return get_obs(state, params), state, reward, done, {}


def reset(rng_input, params):
    """ Reset environment state. """
    optimal_no_cost = ((1 - params["deterministic"])
                        * (1 - 1 / params["size"]) ** (params["size"] - 1)
                        + params["deterministic"] * 1.)
    optimal_return = optimal_no_cost - params["unscaled_move_cost"]
    state = {"row": 0,
             "column": 0,
             "bad_episode": False,
             "total_bad_episodes": 0,
             "denoised_return": 0,
             "optimal_no_cost": optimal_no_cost,
             "optimal_return": optimal_return,
             "terminal": False}
    return get_obs(state, params), state


def get_obs(state, params):
    """ Return observation from raw state trafo. """
    obs_end = jnp.zeros(shape=(params["size"], params["size"]),
                        dtype=jnp.float32)
    end_cond = state["row"] >= params["size"]
    obs_upd = jax.ops.index_update(obs_end, jax.ops.index[state["row"],
                                                          state["column"]], 1.)
    return (1 - end_cond) * obs_end + end_cond * obs_upd


reset_deep_sea = jit(reset, static_argnums=(1,))
step_deep_sea = jit(step, static_argnums=(1,))

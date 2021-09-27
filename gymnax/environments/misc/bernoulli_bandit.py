import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict


params_bandit = FrozenDict({"sample_probs": jnp.array([0.1, 0.9]),
                            "max_steps": 100})


def step(rng_input, params, state, action):
    """ Sample bernoulli reward, increase counter, construct input. """
    timestep = state[2] + 1
    done = (timestep >= params["max_steps"])
    reward = jax.random.bernoulli(rng_input, state[action]).astype(jnp.int32)
    obs = get_obs(reward, action, time, params)
    state = jax.ops.index_update(state, 2, time)
    return obs, state, reward, done, {}


def reset(rng_input, params):
    """ Reset the Bernoulli bandit. Resample arm identities. """
    # Sample reward function + construct state as concat with timestamp
    p1 = jax.random.choice(rng_input, params["sample_probs"],
                           shape=(1,)).squeeze()
    # State representation: Mean reward a1, Mean reward a2, t
    state = jnp.stack([p1, 1 - p1, 0])
    return get_obs(0, 0, 0, params), state


def get_obs(reward, action, time, params):
    """ Concatenate reward, one-hot action and time stamp. """
    action_one_hot = jax.nn.one_hot(action, 2).squeeze()
    return jnp.hstack([reward, action_one_hot, time])


reset_bandit = jit(reset, static_argnums=(1,))
step_bandit = jit(step, static_argnums=(1,))

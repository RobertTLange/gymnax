import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


def init_buffer(state_template, obs_template, action_template, capacity):
    """ Initialize jnp arrays based on shape of obs and capacity. """
    # Get shape of obs, state, action from template arrays
    state_shape = list(state_template.shape)
    obs_shape = list(obs_template.shape)
    action_shape = list(action_template.shape)
    buffer = {}

    # Initialize buffers for s_t, s_t_1, o_t, o_t_1, a_t, r_t_1, done_t
    buffer["state"] = jnp.zeros([capacity] + state_shape)
    buffer["next_state"] = jnp.zeros([capacity] + state_shape)
    buffer["obs"] = jnp.zeros([capacity] + obs_shape)
    buffer["next_obs"] = jnp.zeros([capacity] + obs_shape)
    buffer["action"] = jnp.zeros([capacity] + action_shape)
    buffer["reward"] =jnp.zeros(capacity)
    buffer["done"] = jnp.zeros(capacity)

    # Use pointer in buffer to overwrite entries once buffer runs over
    buffer["capacity"] = capacity
    buffer["pointer"] = 0
    buffer["total_transitions"] = 0
    return buffer


@jax.jit
def push_to_buffer(buffer, state, next_state, obs, next_obs,
                   action, reward, done):
    """ Store transition tuple data at step pointer location. """
    buffer["state"] = jax.ops.index_update(buffer["state"],
                        jax.ops.index[buffer["pointer"], :], state)
    buffer["next_state"] = jax.ops.index_update(buffer["next_state"],
                        jax.ops.index[buffer["pointer"], :], next_state)
    buffer["obs"] = jax.ops.index_update(buffer["obs"],
                      jax.ops.index[buffer["pointer"], :], obs)
    buffer["next_obs"] = jax.ops.index_update(buffer["next_obs"],
                           jax.ops.index[buffer["pointer"], :], next_obs)
    buffer["action"] = jax.ops.index_update(buffer["action"],
                         jax.ops.index[buffer["pointer"], :], action)
    buffer["reward"] = jax.ops.index_update(buffer["reward"],
                         jax.ops.index[buffer["pointer"]], reward)
    buffer["done"] = jax.ops.index_update(buffer["done"],
                        jax.ops.index[buffer["pointer"]], done)

    # Update the buffer pointer, reset once capacity is full - overwrite
    buffer["pointer"] += 1
    buffer["pointer"] = buffer["pointer"] % buffer["capacity"]
    buffer["total_transitions"] = jnp.minimum(buffer["total_transitions"]
                                              + 1, buffer["capacity"])
    return buffer


@partial(jit, static_argnums=(2,))
def sample_from_buffer(key, buffer, batch_size):
    """ Sample a batch from buffer: (idx, o_t, o_t_1, a_t, r_t_1, done_t) """
    sample_idx = jax.random.randint(key, shape=(batch_size, ),
                                    minval=0,
                                    maxval=buffer["total_transitions"])
    # TODO: Figure how to do sampling without replacement without
    # random.choice + jnp.arange since arange varies in shape = no jit
    #valid_idx = jnp.arange(start=0, stop=buffer["total_transitions"],
    #                       step=1)
    #sample_idx = jax.random.choice(key, valid_idx, shape=(batch_size, ),
    #                               replace=False)
    batch_out = {}
    batch_out["idx"] = sample_idx
    batch_out["obs"] = jnp.take(buffer["obs"], sample_idx, axis=0)
    batch_out["next_obs"] = jnp.take(buffer["next_obs"], sample_idx, axis=0)
    batch_out["action"] = jnp.take(buffer["action"], sample_idx, axis=0)
    batch_out["reward"] = jnp.take(buffer["reward"], sample_idx, axis=0)
    batch_out["done"] = jnp.take(buffer["done"], sample_idx, axis=0)
    return batch_out

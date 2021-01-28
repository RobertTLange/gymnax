import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


class ReplayBuffer(object):
    """A JAX-based replay buffer. JIT-through sampling and storage calls. """
    def __init__(self, state_template, obs_template, action_template,
                 capacity):
        """ Initialize jnp arrays based on shape of obs and capacity. """
        # Get shape of obs, state, action from template arrays
        self.state_shape = list(state_template.shape)
        self.obs_shape = list(obs_template.shape)
        self.action_shape = list(action_template.shape)

        # Initialize buffers for s_t, s_t_1, o_t, o_t_1, a_t, r_t_1, done_t
        self.state_buffer = jnp.zeros([capacity] + self.state_shape)
        self.next_state_buffer = jnp.zeros([capacity] + self.state_shape)
        self.obs_buffer = jnp.zeros([capacity] + self.obs_shape)
        self.next_obs_buffer = jnp.zeros([capacity] + self.obs_shape)
        self.action_buffer = jnp.zeros([capacity] + self.action_shape)
        self.reward_buffer =jnp.zeros(capacity)
        self.done_buffer = jnp.zeros(capacity)

        # Use pointer in buffer to overwrite entries once buffer runs over
        self.capacity = capacity
        self.buffer_pointer = 0
        self.total_transitions = 0

    @partial(jit, static_argnums=(0,))
    def push(self, state, next_state, obs, next_obs, action, reward, done):
        """ Store transition tuple data at step pointer location. """
        self.state_buffer = jax.ops.index_update(self.state_buffer,
                            jax.ops.index[self.buffer_pointer, :], state)
        self.next_state_buffer = jax.ops.index_update(self.next_state_buffer,
                            jax.ops.index[self.buffer_pointer, :], next_state)
        self.obs_buffer = jax.ops.index_update(self.obs_buffer,
                          jax.ops.index[self.buffer_pointer, :], obs)
        self.next_obs_buffer = jax.ops.index_update(self.next_obs_buffer,
                               jax.ops.index[self.buffer_pointer, :], next_obs)
        self.action_buffer = jax.ops.index_update(self.action_buffer,
                             jax.ops.index[self.buffer_pointer, :], action)
        self.reward_buffer = jax.ops.index_update(self.reward_buffer,
                             jax.ops.index[self.buffer_pointer], reward)
        self.done_buffer = jax.ops.index_update(self.done_buffer,
                            jax.ops.index[self.buffer_pointer], done)

        # Update the buffer pointer, reset once capacity is full - overwrite
        self.buffer_pointer += 1
        self.total_transitions = jnp.minimum(self.total_transitions + 1,
                                             self.capacity)
        self.buffer_pointer = self.buffer_pointer % self.capacity

    @partial(jit, static_argnums=(0, 2,))
    def sample(self, key, batch_size):
        """ Sample a batch from buffer: (o_t, o_t_1, a_t, r_t_1, done_t) """
        valid_idx = jnp.arange(start=0, stop=self.total_transitions, step=1)
        sample_idx = jax.random.choice(key, valid_idx, shape=(batch_size, ),
                                       replace=False)
        obs_batch = jnp.take(self.obs_buffer, sample_idx, axis=0)
        next_obs_batch = jnp.take(self.next_obs_buffer, sample_idx, axis=0)
        action_batch = jnp.take(self.action_buffer, sample_idx, axis=0)
        reward_batch = jnp.take(self.reward_buffer, sample_idx, axis=0)
        done_batch = jnp.take(self.done_buffer, sample_idx, axis=0)
        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch

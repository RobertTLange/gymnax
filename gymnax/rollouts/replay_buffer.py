import jax
import jax.numpy as jnp


class ReplayBuffer(object):
    """A JAX-based replay buffer. JIT-through sampling and storage calls. """
    def __init__(self, action_template, obs_template, capacity):
        """ Initialize jnp arrays based on shape of obs and capacity. """
        self.obs_shape = list(obs_template.shape)
        self.action_shape = list(action_template.shape)
        self.obs_buffer = jnp.zeros([capacity] + self.obs_shape)
        self.next_obs_buffer = jnp.zeros([capacity] + self.obs_shape)
        self.action_buffer = jnp.zeros([capacity] + self.action_shape)
        self.reward_buffer =jnp.zeros(capacity)
        self.done_buffer = jnp.zeros(capacity)
        # Use pointer in buffer to overwrite entries once buffer runs over
        self.buffer_pointer = 0

    def push(self, state, next_state, obs, next_obs, action, reward, done):
        raise NotImplementedError

    def sample(self, batch_size):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size))
        return (jnp.stack(obs_tm1), jnp.asarray(a_tm1), jnp.asarray(r_t),
                jnp.asarray(discount_t) * train_config["discount_factor"],
                jnp.stack(obs_t))

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

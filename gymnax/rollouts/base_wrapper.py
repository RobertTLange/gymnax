import jax
import jax.numpy as jnp
from jax import lax, jit, vmap


class BaseWrapper(object):
    """ Base wrapper for episode rollouts. """
    def __init__(self, policy, step, reset,
             env_params, max_steps_in_episode):
        self.policy = policy
        self.step = step
        self.reset = reset
        self.env_params = env_params
        self.max_steps_in_episode = max_steps_in_episode

    def policy_step(self, state_input, tmp):
        """ lax.scan compatible step transition in JAX env. """
        rng, obs, state, policy_params, env_params = state_input
        rng, rng_input = jax.random.split(rng)
        action = self.policy(policy_params, obs)
        next_o, next_s, reward, done, _ = self.step(rng_input, env_params,
                                                    state, action)
        carry, y = [rng, next_o.squeeze(), next_s.squeeze(),
                    policy_params, env_params], [reward]
        return carry, y

    def lax_rollout(self, rng_input, policy_params, env_params,
                    max_steps_in_episode):
        """ Rollout a pendulum episode with lax.scan. """
        obs, state = self.reset(rng_input, env_params)
        scan_out1, scan_out2 = lax.scan(
                            self.policy_step,
                            [rng_input, obs, state,
                             policy_params, env_params],
                            [jnp.zeros(max_steps_in_episode)])
        return scan_out1, jnp.array(scan_out2)

    def episode_rollout(self, rng_input, policy_params):
        """ Jitted episode rollout. """
        ep_rollout = jit(self.lax_rollout, static_argnums=(3))
        trace, reward = ep_rollout(rng_input, policy_params,
                                   self.env_params,
                                   self.max_steps_in_episode)
        return trace, reward

    def batch_rollout(self, rng_input, policy_params):
        """ vmap across keys used to initialize an episode. """
        batch_rollout = jit(vmap(self.lax_rollout,
                                 in_axes=(0, None, None, None), out_axes=0),
                            static_argnums=(3))
        traces, rewards = batch_rollout(rng_input, policy_params,
                                        self.env_params,
                                        self.max_steps_in_episode)
        return traces, rewards

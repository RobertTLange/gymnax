import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial


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

    @partial(jit, static_argnums=(0, 4))
    def lax_rollout(self, rng_input, policy_params, env_params,
                    max_steps_in_episode):
        """ Rollout a gymnax episode with lax.scan. """
        obs, state = self.reset(rng_input, env_params)
        scan_out1, scan_out2 = lax.scan(
                            self.policy_step,
                            [rng_input, obs, state,
                             policy_params, env_params],
                            [jnp.zeros(max_steps_in_episode)])
        return scan_out1, jnp.array(scan_out2).squeeze()

    @partial(jit, static_argnums=(0, 4))
    def vmap_rollout(self, rng_input, policy_params, env_params,
                     max_steps_in_episode):
        """ Jit + vmap wrapper around scanned episode rollout. """
        rollout_map = vmap(self.lax_rollout,
                           in_axes=(0, None, None, None), out_axes=0)
        traces, rewards = rollout_map(rng_input, policy_params,
                                      self.env_params,
                                      self.max_steps_in_episode)
        return traces, rewards

    def episode_rollout(self, rng_input, policy_params):
        """ Jitted episode rollout for single episode. """
        trace, reward = self.lax_rollout(rng_input, policy_params,
                                         self.env_params,
                                         self.max_steps_in_episode)
        return trace, reward

    def batch_rollout(self, rng_input, policy_params):
        """ Vmapped episode rollout for set of episodes. """
        traces, rewards = self.vmap_rollout(rng_input, policy_params,
                                            self.env_params,
                                            self.max_steps_in_episode)
        return traces, rewards

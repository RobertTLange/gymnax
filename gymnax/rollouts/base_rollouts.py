import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial


class BaseRollouts(object):
    """ Base wrapper for episode rollouts. """
    def __init__(self, step, reset, env_params):
        self.step = step
        self.reset = reset
        self.env_params = env_params
        self.max_steps_in_episode = env_params["max_steps_in_episode"]

    def action_selection(self, rng, agent_params, obs):
        """ Compute action to be executed in environment. """
        raise NotImplementedError

    def prepare_experience(self, env_output, net_output):
        """ Prepare the generated data (net/env) to be stored in a buffer. """
        raise NotImplementedError

    def store_experience(self, step_experience):
        """ Store the transition data (net + env) in a buffer. """
        raise NotImplementedError

    def update_learner(self, agent_params):
        """ Perform an update to the parameters of the learner. """
        raise NotImplementedError

    def perform_transition(self, rng, env_params, state, action):
        """ Perform the step transition in the environment. """
        next_obs, next_state, reward, done, _ = self.step(rng, env_params,
                                                          state, action)
        return next_obs, next_state, reward, done, _

    def actor_learner_step(self, carry_input, tmp):
        """ lax.scan compatible step transition in JAX env.
            This implements an alternating actor-learner paradigm for
            each step transition in the environment. Rewrite for case of
            On-Policy methods and update at end of episode.
        """
        # 0. Unpack carry, split rng key for action selection + transition
        rng, obs, state, agent_params, env_params = carry_input
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # 1. Perform action selection using actor NN
        action, net_output = self.action_selection(rng_act, agent_params, obs)

        # 2. Perform the step transition in the environment & format env output
        next_obs, next_state, reward, done, _ = self.perform_transition(
                                    rng_step, env_params, state, action)
        env_output = (state, next_state, obs, next_obs, action, reward, done)

        # 3. Prepare gathered info from transition (env + net) [keep state info]
        step_experience = self.prepare_experience(env_output, net_output)

        # 4. Store the transition in a transition buffer
        self.store_experience(step_experience)

        # 5. Update the learner by e.g. performing some SGD update
        self.update_learner(agent_params)

        # 6. Collect all relevant data for next actor-learner-step
        carry, y = [rng, next_obs.squeeze(), next_state.squeeze(),
                    agent_params, env_params], [reward]
        return carry, y

    @partial(jit, static_argnums=(0, 4))
    def lax_rollout(self, rng_input, agent_params, env_params,
                    max_steps_in_episode):
        """ Rollout a gymnax episode with lax.scan. """
        obs, state = self.reset(rng_input, env_params)
        scan_out1, scan_out2 = lax.scan(
                            self.actor_learner_step,
                            [rng_input, obs, state,
                             agent_params, env_params],
                            [jnp.zeros(max_steps_in_episode)])
        return scan_out1, jnp.array(scan_out2).squeeze()

    @partial(jit, static_argnums=(0, 4))
    def vmap_rollout(self, rng_input, agent_params, env_params,
                     max_steps_in_episode):
        """ Jit + vmap wrapper around scanned episode rollout. """
        rollout_map = vmap(self.lax_rollout,
                           in_axes=(0, None, None, None), out_axes=0)
        traces, rewards = rollout_map(rng_input, agent_params,
                                      self.env_params,
                                      self.max_steps_in_episode)
        return traces, rewards

    def episode_rollout(self, rng_input, agent_params):
        """ Jitted episode rollout for single episode. """
        trace, reward = self.lax_rollout(rng_input, agent_params,
                                         self.env_params,
                                         self.max_steps_in_episode)
        return trace, reward

    def batch_rollout(self, rng_input, agent_params):
        """ Vmapped episode rollout for set of episodes. """
        traces, rewards = self.vmap_rollout(rng_input, agent_params,
                                            self.env_params,
                                            self.max_steps_in_episode)
        return traces, rewards

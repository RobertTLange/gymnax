import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial


class BaseDojo(object):
    """ Base wrapper for episode rollouts. """
    def __init__(self, step, reset, env_params):
        self.step = step
        self.reset = reset
        self.env_params = env_params

    def action_selection(self, key, obs, agent_params, actor_state):
        """ Compute action to be executed in environment. """
        raise NotImplementedError

    def prepare_experience(self, env_output, actor_state):
        """ Prepare the generated data (net/env) to be stored in a buffer. """
        raise NotImplementedError

    def store_experience(self, step_experience):
        """ Store the transition data (net + env) in a buffer. """
        raise NotImplementedError

    def update_learner(self, key, agent_params, learner_state):
        """ Perform an update to the parameters of the learner. """
        raise NotImplementedError

    def init_learner_state(self, agent_params):
        """ Initialize the state of the learner (e.g. optimizer). """
        raise NotImplementedError

    def init_actor_state(self):
        """ Initialize the state of the actor (e.g. for exploration). """
        raise NotImplementedError

    def init_dojo(self, agent_params=None):
        """ Initialize the rollout collector/learning dojo. """
        self.agent_params = agent_params
        self.learner_state = self.init_learner_state(agent_params)
        self.actor_state = self.init_actor_state()

    def perform_transition(self, key, env_params, state, action):
        """ Perform the step transition in the environment. """
        next_obs, next_state, reward, done, _ = self.step(key, env_params,
                                                          state, action)
        return next_obs, next_state, reward, done, _

    def actor_learner_step(self, carry_input, tmp):
        """ lax.scan compatible step transition in JAX env.
            This implements an alternating actor-learner paradigm for
            each step transition in the environment. Rewrite for case of
            On-Policy methods and update at end of episode.
        """
        # 0. Unpack carry, split rng key for action selection + transition
        rng, obs, state, env_params = carry_input[0:4]
        agent_params, actor_state, learner_state = carry_input[4:7]
        rng, key_act, key_step, key_learn, key_reset = jax.random.split(rng, 5)

        # 1. Perform action selection using actor NN
        action, actor_state = self.action_selection(key_act, obs,
                                                    agent_params,
                                                    actor_state)

        # 2. Perform step transition in the environment & format env output
        next_obs, next_state, reward, done, _ = self.perform_transition(
                                    key_step, env_params, state, action)
        env_output = (state, next_state, obs, next_obs,
                      action, reward, done)

        # 3. Prepare info from transition (env + net) [keep state info]
        step_experience = self.prepare_experience(env_output, actor_state)

        # 4. Store the transition in a transition buffer
        self.store_experience(step_experience)

        # 5. Update the learner by e.g. performing some SGD update
        agent_params, learner_state = self.update_learner(key_learn,
                                                          agent_params,
                                                          learner_state)

        # 6. Auto-reset environment and use obs/state if episode terminated
        obs_reset, state_reset = self.reset(key_reset, env_params)
        next_obs = done * obs_reset + (1 - done) * next_obs
        next_state = done * state_reset + (1 - done) * next_state

        # 7. Collect all relevant data for next actor-learner-step
        carry, y = ([rng, next_obs.squeeze(), next_state.squeeze(),
                     env_params, agent_params, actor_state, learner_state],
                    [reward])
        return carry, y

    @partial(jit, static_argnums=(0, 3))
    def lax_rollout(self, key_input, env_params, num_steps,
                    agent_params, actor_state, learner_state):
        """ Rollout a gymnax episode with lax.scan. """
        # Reset the environment once at beginning of step rollout
        obs, state = self.reset(key_input, env_params)
        # Rollout the steps in the environment (w. potential resets)
        scan_out1, scan_out2 = lax.scan(
                            self.actor_learner_step,
                            [key_input, obs, state, env_params,
                             agent_params, actor_state, learner_state],
                            [jnp.zeros(num_steps)])
        return scan_out1, jnp.array(scan_out2).squeeze()

    @partial(jit, static_argnums=(0, 3))
    def vmap_rollout(self, key_input, env_params, num_steps,
                     agent_params, actor_state, learner_state):
        """ Jit + vmap wrapper around scanned episode rollout. """
        rollout_map = vmap(self.lax_rollout,
                           in_axes=(0, None, None, None, None, None),
                           out_axes=0)
        traces, rewards = rollout_map(key_input, env_params,
                                      num_steps, agent_params,
                                      actor_state, learner_state)
        return traces, rewards

    def steps_rollout(self, key_rollout, num_steps, agent_params=None):
        """ Jitted episode rollout for single episode. """
        try:
            # Different cases: agent_params explicitly supplied/when not
            if agent_params is None:
                trace, reward = self.lax_rollout(key_rollout,
                                                 self.env_params,
                                                 num_steps,
                                                 self.agent_params,
                                                 self.actor_state,
                                                 self.learner_state)
            else:
                trace, reward = self.lax_rollout(key_rollout,
                                                 self.env_params,
                                                 num_steps,
                                                 agent_params,
                                                 self.actor_state,
                                                 self.learner_state)
        except AttributeError as err:
            raise AttributeError(f"{err}. Please initialize the "
                                  "agent's parameters and the states "
                                  "of the actor and learner.")
        return trace, reward

    def batch_rollout(self, key_rollout, num_steps, agent_params=None):
        """ Vmapped episode rollout for set of episodes. """
        try:
            # Different cases: agent_params explicitly supplied/when not
            if agent_params is None:
                traces, rewards = self.vmap_rollout(key_rollout,
                                                    self.env_params,
                                                    num_steps,
                                                    self.agent_params,
                                                    self.actor_state,
                                                    self.learner_state)
            else:
                traces, rewards = self.vmap_rollout(key_rollout,
                                                    self.env_params,
                                                    num_steps,
                                                    agent_params,
                                                    self.actor_state,
                                                    self.learner_state)
        except AttributeError as err:
            raise AttributeError(f"{err}. Please initialize the "
                                  "agent params and/or actor/learner states.")
        return traces, rewards

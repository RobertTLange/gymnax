import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from gymnax.rollouts.base_rollouts import BaseRollouts


class InterleavedRollouts(BaseRollouts):
    """ Interleaved rollouts of acting-learning as in DQN-style algorithms. """
    def __init__(self, agent, buffer, step, reset, env_params):
        BaseRollouts.__init__(self, step, reset, env_params)
        self.agent = agent
        self.buffer = buffer

    def action_selection(self, rng, agent_params, obs):
        """ Compute action to be executed in environment. """
        action = self.agent.actor_step(agent_params, obs)
        return action

    def prepare_experience(self, env_output, net_output):
        """ Prepare the generated data (net/env) to be stored in a buffer. """
        raise NotImplementedError

    def store_experience(self, step_experience):
        """ Store the transition data (net + env) in a buffer. """
        self.buffer.push(...)

    def update_learner(self, agent_params):
        """ Perform an update to the parameters of the learner. """
        agent_params = self.agent.learner_step()
        return agent_params

    def perform_transition(self, rng, env_params, state, action):
        """ Perform the step transition in the environment. """
        next_o, next_s, reward, done, _ = self.step(rng, env_params,
                                                    state, action)
        return next_o, next_s, reward, done, _

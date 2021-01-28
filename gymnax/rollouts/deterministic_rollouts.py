import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from gymnax.rollouts.base_rollouts import BaseRollouts


class DeterministicRollouts(BaseRollouts):
    """ Deterministic episode rollouts without learning (e.g. in neuroevo). """
    def __init__(self, policy, step, reset, env_params):
        BaseRollouts.__init__(self, step, reset, env_params)
        self.policy = policy

    def action_selection(self, key, obs, agent_params, actor_state):
        """ Compute action to be executed in environment. """
        action = self.policy(agent_params, obs)
        return action, None

    def prepare_experience(self, env_output, actor_state):
        """ Prepare the generated data (net/env) to be stored in a buffer. """
        return None

    def store_experience(self, step_experience):
        """ Store the transition data (net + env) in a buffer. """
        return None

    def update_learner(self, agent_params, learner_state):
        """ Perform an update to the parameters of the learner. """
        return agent_params, None

    def init_learner_state(self, agent_params):
        """ Initialize the state of the learner (e.g. optimizer). """
        return None

    def init_actor_state(self):
        """ Initialize the state of the actor (e.g. for exploration). """
        return None

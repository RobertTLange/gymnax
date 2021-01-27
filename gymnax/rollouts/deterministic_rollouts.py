import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from gymnax.rollouts.base_rollouts import BaseRollouts


class DeterministicRollouts(BaseRollouts):
    """ Deterministic episode rollouts without learning.
        As for example used in neuroevolution experiments.
    """
    def __init__(self, policy, step, reset, env_params):
        BaseRollouts.__init__(self, step, reset, env_params)
        self.policy = policy

    def action_selection(self, rng, agent_params, obs):
        """ Compute action to be executed in environment. """
        action = self.policy(agent_params, obs)
        return action, None

    def prepare_experience(self, env_output, net_output):
        """ Prepare the generated data (net/env) to be stored in a buffer. """
        return None

    def store_experience(self, step_experience):
        """ Store the transition data (net + env) in a buffer. """
        return None

    def update_learner(self, agent_params):
        """ Perform an update to the parameters of the learner. """
        return agent_params

    def perform_transition(self, rng, env_params, state, action):
        """ Perform the step transition in the environment. """
        next_o, next_s, reward, done, _ = self.step(rng, env_params,
                                                    state, action)
        return next_o, next_s, reward, done, _

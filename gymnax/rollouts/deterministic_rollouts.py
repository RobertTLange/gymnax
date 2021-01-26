import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from gymnax.rollouts.base_rollouts import BaseRollouts


class DeterministicRollouts(BaseRollouts):
    """ Base wrapper for episode rollouts. """
    def __init__(self, policy, step, reset, env_params):
        BaseRollouts.__init__(self, policy, step, reset, env_params)

    def action_selection(self, rng, policy_params, obs):
        """ Compute action to be executed in environment. """
        action = self.policy(policy_params, obs)
        return action

    def store_transition(self, obs, state, reward, done):
        """ Store the transition in a buffer. """
        return None

    def update_learner(self, policy_params, buffer):
        """ Perform an update to the parameters of the learner. """
        return policy_params

    def perform_transition(self, rng, env_params, state, action):
        """ Perform the step transition in the environment. """
        next_o, next_s, reward, done, _ = self.step(rng, env_params,
                                                    state, action)
        return next_o, next_s, reward, done, _

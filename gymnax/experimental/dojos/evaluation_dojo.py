import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from .base_dojo import BaseDojo


class EvaluationDojo(BaseDojo):
    """Evaluation episode rollouts w\o learning (e.g. neuroevo/RL testing)."""

    def __init__(self, agent, step, reset, env_params):
        BaseDojo.__init__(self, step, reset, env_params)
        self.agent = agent

    def action_selection(self, key, obs, agent_params, actor_state):
        """Compute action to be executed in environment."""
        action, actor_state = self.agent.actor_step(key, agent_params, obs, actor_state)
        return action, actor_state

    def prepare_experience(self, env_output, actor_state):
        """Prepare generated data (net/env) to be stored in buffer."""
        return None

    def store_experience(self, step_experience):
        """Store the transition data (net + env) in a buffer."""
        return None

    def update_learner(self, key, agent_params, learner_state):
        """Perform an update to the parameters of the learner."""
        return agent_params, None

    def init_learner_state(self, agent_params):
        """Initialize the state of the learner (e.g. optimizer)."""
        return None

    def init_actor_state(self):
        """Initialize the state of the actor (e.g. exploration)."""
        return self.agent.init_actor_state(evaluate=True)

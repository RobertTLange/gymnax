class MinimalEvaluationAgent:
    """A Minimal Wrapper for an evaluation agent."""

    def __init__(self, policy):
        """Init all key features of the agent. E.g. this may include:
        - Policy function network forward function (assumes key input)
        - Exploitation schedule to use in evaluation
        - Here: Deterministic Agent - but could also be stochastic!
        """
        self.policy = policy

    def actor_step(self, key, agent_params, obs, actor_state):
        """Policy forward pass + return action and new state."""
        action = self.policy(key, agent_params, obs)
        return action, actor_state

    def init_actor_state(self, evaluate=False):
        return None


class MinimalInterleavedAgent:
    """A Minimal Wrapper for an agent that interleaves steps with updates."""

    def __init__(self, policy):
        """Init all key features of the agent. E.g. this may include:
        - Policy/Value function network forward function
        - Optimizer to use in learner_step = Use optax!
        - Exploration schedule to use in actor_step
        """
        self.policy = policy

    def actor_step(self, key, agent_params, obs, actor_state):
        """Policy forward pass + return action and new state."""
        action = self.policy(key, agent_params, obs)
        return action, actor_state

    def learner_step(self, key, agent_params, learner_state):
        """Update the network params + return new state (e.g. of opt)."""
        return agent_params, learner_state

    def init_learner_state(self, agent_params):
        return None

    def init_actor_state(self, evaluate=False):
        return None

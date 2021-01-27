import jax
import jax.numpy as jnp
import optax
import rlax
import haiku as hk
from haiku import nets

import time
import collections
import random
import gymnax



class DQN:
    """A simple DQN agent."""
    def __init__(self, observation_spec, action_spec, epsilon_cfg,
                 target_period, learning_rate):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._target_period = target_period
        # Neural net and optimiser.
        self._network = build_network(action_spec.num_values)
        self._optimizer = optax.adam(learning_rate)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        # Jitting for speed.
        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initial_params(self, key):
        sample_input = self._observation_spec.generate_value()
        sample_input = jnp.expand_dims(sample_input, 0)
        online_params = self._network.init(key, sample_input)
        return Params(online_params, online_params)

    def initial_actor_state(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(actor_count)

    def initial_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        return LearnerState(learner_count, opt_state)

    def actor_step(self, params, env_output, actor_state, key, evaluation):
        obs = jnp.expand_dims(env_output.observation, 0)  # add dummy batch
        q = self._network.apply(params.online, obs)[0]    # remove dummy batch
        epsilon = self._epsilon_by_frame(actor_state.count)
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        a = jax.lax.select(evaluation, eval_a, train_a)
        return (ActorOutput(actions=a, q_values=q),
                ActorState(actor_state.count + 1))

    def learner_step(self, params, data, learner_state, unused_key):
        target_params = rlax.periodic_update(
            params.online, params.target,
            learner_state.count, self._target_period)
        dloss_dtheta = jax.grad(self._loss)(params.online, target_params, *data)
        updates, opt_state = self._optimizer.update(dloss_dtheta,
                                                    learner_state.opt_state)
        online_params = optax.apply_updates(params.online, updates)
        return (Params(online_params, target_params),
                LearnerState(learner_state.count + 1, opt_state))

    def _loss(self, online_params, target_params,
            obs_tm1, a_tm1, r_t, discount_t, obs_t):
        q_tm1 = self._network.apply(online_params, obs_tm1)
        q_t_val = self._network.apply(target_params, obs_t)
        q_t_select = self._network.apply(online_params, obs_t)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t,
                                q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))


def run_loop(agent, environment, accumulator, seed,
             batch_size, train_episodes, evaluate_every, eval_episodes):
    """A simple run loop for examples of reinforcement learning with rlax."""

    # Init agent.
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(next(rng))
    learner_state = agent.initial_learner_state(params)

    print(f"Training agent for {train_episodes} episodes")
    for episode in range(train_episodes):
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        accumulator.push(timestep, None)
        actor_state = agent.initial_actor_state()

        while not timestep.last():

            # Acting.
            actor_output, actor_state = agent.actor_step(
                params, timestep, actor_state, next(rng), evaluation=False)

            # Agent-environment interaction.
            timestep = environment.step(int(actor_output.actions))

            # Accumulate experience.
            accumulator.push(timestep, actor_output.actions)

            # Learning.
            if accumulator.is_ready(batch_size):
                params, learner_state = agent.learner_step(
                    params, accumulator.sample(batch_size),
                    learner_state, next(rng))

        # Evaluation.
        if not episode % evaluate_every:
            returns = 0.
            for _ in range(eval_episodes):
                timestep = environment.reset()
                actor_state = agent.initial_actor_state()

                while not timestep.last():
                    actor_output, actor_state = agent.actor_step(
                      params, timestep, actor_state, next(rng), evaluation=True)
                    timestep = environment.step(int(actor_output.actions))
                    returns += timestep.reward

            avg_returns = returns / eval_episodes
            print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")


def build_network(num_actions: int) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""
    def q(obs):
        network = hk.Sequential(
            [hk.Flatten(),
             nets.MLP([train_config["hidden_units"], num_actions])])
        return network(obs)
    return hk.without_apply_rng(hk.transform(q, apply_rng=True))


def main(train_config):
    env = catch.Catch(seed=train_config["seed"])
    epsilon_cfg = dict(init_value=train_config["epsilon_begin"],
                       end_value=train_config["epsilon_end"],
                       transition_steps=train_config["epsilon_steps"],
                       power=1.)
    agent = DQN(observation_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                epsilon_cfg=epsilon_cfg,
                target_period=train_config["target_period"],
                learning_rate=train_config["learning_rate"])

    accumulator = ReplayBuffer(train_config["replay_capacity"])
    run_loop(agent=agent,
             environment=env,
             accumulator=accumulator,
             seed=train_config["seed"],
             batch_size=train_config["batch_size"],
             train_episodes=train_config["train_episodes"],
             evaluate_every=train_config["evaluate_every"],
             eval_episodes=train_config["eval_episodes"])


Params = collections.namedtuple("Params", "online target")
ActorState = collections.namedtuple("ActorState", "count")
ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")
Data = collections.namedtuple("Data", "obs_tm1 a_tm1 r_t discount_t obs_t")


train_config = {"seed": 42,
                "train_episodes": 301,
                "batch_size": 32,
                "target_period": 50,
                "replay_capacity": 2000,
                "hidden_units": 50,
                "epsilon_begin": 1.,
                "epsilon_end": 0.01,
                "epsilon_steps": 1000,
                "discount_factor": 0.99,
                "learning_rate": 0.005,
                "eval_episodes": 100,
                "evaluate_every": 50}

if __name__ == "__main__":
    start_t = time.time()
    main(train_config)
    stop_t = time.time()
    print("Done with {} episodes after {:.2f} seconds".format(train_config["train_episodes"],
                                                              stop_t - start_t))

import jax
import jax.numpy as jnp
import haiku as hk
from haiku import nets
import optax, rlax, gymnax
import collections, time
from gymnax.dojos import InterleavedDojo, EvaluationDojo
from gymnax.utils import init_buffer, push_buffer, sample_buffer
from catch_dqn_rlax_bsuite import build_network, Params, ActorOutput

ActorState = collections.namedtuple("ActorState", "count evaluation")
LearnerState = collections.namedtuple("LearnerState", "count opt_state discount_factor")


class DQN:
    """A simple Double DQN agent."""

    def __init__(
        self,
        obs_template,
        num_actions,
        epsilon_cfg,
        target_period,
        learning_rate,
        discount_factor,
    ):
        self._obs_template = obs_template
        self._num_actions = num_actions
        self._target_period = target_period
        self._network = build_network(num_actions)
        self._optimizer = optax.adam(learning_rate)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        self._discount_factor = discount_factor

    def initial_params(self, key):
        sample_input = jnp.expand_dims(self._obs_template, 0)
        online_params = self._network.init(key, sample_input)
        return Params(online_params, online_params)

    def init_actor_state(self, evaluate=False):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(actor_count, evaluate)

    def init_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        return LearnerState(learner_count, opt_state, self._discount_factor)

    def actor_step(self, key, params, obs, actor_state):
        obs = jnp.expand_dims(obs, 0)
        q = self._network.apply(params.online, obs)[0]
        epsilon = self._epsilon_by_frame(actor_state.count)
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        a = jax.lax.select(actor_state.evaluation, eval_a, train_a)
        return (a, ActorState(actor_state.count + 1, bool(0)))

    def learner_step(self, key, params, learner_state, data):
        target_params = rlax.periodic_update(
            params.online, params.target, learner_state.count, self._target_period
        )
        discount = (1 - data["done"]) * learner_state.discount_factor
        dloss_dtheta = jax.grad(self._loss)(
            params.online,
            target_params,
            data["obs"],
            data["action"],
            data["reward"],
            discount,
            data["next_obs"],
        )
        updates, opt_state = self._optimizer.update(
            dloss_dtheta, learner_state.opt_state
        )
        online_params = optax.apply_updates(params.online, updates)
        return (
            Params(online_params, target_params),
            LearnerState(learner_state.count + 1, opt_state, self._discount_factor),
        )

    def _loss(
        self, online_params, target_params, obs_tm1, a_tm1, r_t, discount_t, obs_t
    ):
        q_tm1 = self._network.apply(online_params, obs_tm1)
        q_t_val = self._network.apply(target_params, obs_t)
        q_t_select = self._network.apply(online_params, obs_t)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(
            q_tm1,
            a_tm1.squeeze().astype(int),
            r_t.squeeze(),
            discount_t.squeeze(),
            q_t_val,
            q_t_select,
        )
        return jnp.mean(rlax.l2_loss(td_error))


def main(train_config):
    rng, reset, step, env_params = gymnax.make("Catch-bsuite")
    rng, key_reset = jax.random.split(rng)
    obs, state = reset(key_reset, env_params)
    action = jnp.array([0])
    buffer = init_buffer(state, obs, action, train_config["replay_capacity"])
    num_actions = 3
    epsilon_cfg = dict(
        init_value=train_config["epsilon_begin"],
        end_value=train_config["epsilon_end"],
        transition_steps=train_config["epsilon_steps"],
        power=1.0,
    )
    rng, rng_net = jax.random.split(rng)
    agent = DQN(
        obs,
        num_actions,
        epsilon_cfg,
        train_config["target_period"],
        train_config["learning_rate"],
        train_config["discount_factor"],
    )
    agent_params = agent.initial_params(rng_net)
    # Interleaved dojo for actor-learner step-updates
    collector = InterleavedDojo(
        agent, buffer, push_buffer, sample_buffer, step, reset, env_params
    )
    collector.init_dojo(agent_params)

    evaluator = EvaluationDojo(agent, step, reset, env_params)
    evaluator.init_dojo()
    for i in range(train_config["train_step_rollouts"]):
        rng, rng_train, rng_eval = jax.random.split(rng, 3)
        rng_evals = jax.random.split(rng_eval, train_config["eval_episodes"])
        trace, reward = collector.steps_rollout(
            rng_train, train_config["steps_per_rollout"]
        )
        trace, reward = evaluator.batch_rollout(rng_evals, 9, collector.agent_params)
        print(jnp.sum(reward, axis=1).mean())


if __name__ == "__main__":
    train_config = {
        "seed": 42,
        "train_step_rollouts": 10,
        "batch_size": 32,
        "target_period": 50,
        "replay_capacity": 2000,
        "hidden_units": 50,
        "epsilon_begin": 1.0,
        "epsilon_end": 0.01,
        "epsilon_steps": 5000,
        "discount_factor": 0.99,
        "learning_rate": 0.005,
        "eval_episodes": 100,
        "evaluate_every": 300,
        "steps_per_rollout": 300,
    }

    # Run the learning loop
    start_t = time.time()
    main(train_config)
    stop_t = time.time()
    print(
        "Done with {} set of {} steps after {:.2f}"
        " seconds".format(
            train_config["train_step_rollouts"],
            train_config["steps_per_rollout"],
            stop_t - start_t,
        )
    )

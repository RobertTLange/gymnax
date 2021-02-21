## TODOs, Notes & Questions
- [ ] Add timeout condition with env_params and t in state!
- [ ] Episode rollout wrapper
    - [x] Deterministic vs. stochastic policy rollout
    - [ ] Q-Learning Wrapper with update after each transition
    - [ ] Policy Gradient Wrapper with update only after transition
    - [ ] Recurrent Wrapper for training LSTM-style policies
- [ ] Add TPU speed tests
- [ ] Add more environments
    - [ ] bsuite classic example
    - [ ] Gridworld/4 rooms
    - [ ] Toy text from gym
    - [ ] Simple bandit environments
- [ ] Add test for transition correctness compared to OpenAI gym
    - [x] Continuous Control
    - [x] Catch Bsuite
- [x] Add backdoor for rendering in OpenAI gym
- [x] Add random policy/sampling for basic rollout
- [ ] Figure out if numerical errors really matter - set JAX precision

- [ ] Documentation
    - [ ] Pypi installation and github workflow for releases
    - [ ] Add state, obs, action space info as in dm_control paper
    - [ ] Connect notebooks with example Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk
    - [ ] Put together a plot of wall-clock time vs performance + visualization of catch game next to it
    - [ ] Add `env_info` dict to all envs - collects action space/observation space
    - Make decision on discount vs done syntax
    - Adopt DQN example from rlax and jit through entire thing
    - Figure out solution for action and observation space
    - Decide what variables to store and provide specialized wrapper on top
        - Disentangled base wrapper from specialized ones
        - Value based: states, rewards, actions, dones
            - Interleave step with update
            - Add buffer system to store transitions?!
        - PG based: log prob pi, entropy, returns
        - ES: only cumulated rewards
        - Recurrent policy rollout wrapper
    - Implement catch environment from bsuite

## 21/02/21 - Check if DQN is actually learning II

- Add auto termination and reset to wrappers
- Add timestep counter to the environments - done if timestep > env_steps
- Figure out what is wrong with acrobot - wrong state definition
- Check whether JAX DQN agent is learning - does not appear to!
- Very hard to debug!!!

## 11/02/21 - Check if DQN is actually learning I
- Put agent/loop into .py script.
- Need to make a decision about the discount settings and what/how to store
    - Rollout steps not episodes
    - How do we deal with discount annealing? and state resets?
    - 'Purify' buffer from transitions after termination - discount = 0

## 10/02/21 - Back at it!
- Problem with env_params that affect the observation shapes. `FrozenDict` non-hashable?!
    - For now pull out of env_params - no longer `static_argnums`
- DQN `_loss` call problem withs chex in rlax double q loss fct
    - Rank compatibility = squeeze inputs
    - Type compatibility = make actions of type int
- Next: figure out evaluation - maybe in separate dojo? Train and test dojos?!

## 02/02/21 - DQN run?! + PlainWrapper rename
- Rename the `DetermisticRollouts` to `EvaluationDojo`.
- Adapt DQN agent class to our setup.
- Problem with catch env frozen dict + static_argnums. Can't have both!

## 01/02/21 - Add learner step and make actor step for DQN!
- Small buffer jitting bug fix with static_argnums.
- Rewrite PlainRollouts to include an agent class.
    - This allows to easily differentiate between types of policies.
    - Test compatibility with evosax examples.
- Need to get key into update learner step - i.e. for sampling from buffer

## 28/01/21 - Continue Replay Buffer Work + DQN setup
- We need an `actorState`, `learnerState` - check if we can jit through named tuples!
- How can we do evaluation episodes?!
    - Use evaluation actor state? And no update of agent params
    - Or separate lax.scan setup?
- Want a function that updates `agent_params` after vmap spits them out
- Problem in buffer sampling with self.total_transitions to sample from and jit
    - Not a problem if resampling is allowed

## 29/01/21 - Jitted Sampling + InterleavedWrapper
- Play around with randint instead of choice - Ultimately rewrite buffer without usage of class instances. There was some leakage of untraced vars coming from self
- Now use 3 separate functions: `init_buffer`, `push_to_buffer`, `sample_from_buffer`
    - `push_to_buffer` takes `step_experience` dict as input and returns update buffer dict
    - 'sample_from_buffer' returns dict of batches with same keys as `step_experience`
    - Dicts allow for max flexibility when adding things like prioritization
    - In future: Make buffer structure more abstract for easy extension
- Go back to interleaved wrapper and add storage of trajectory
    - Include buffer data collection
    - First jitted episode rollout!

## 27/01/21 - Work on stochastic wrapper, value-based wrapper, PG wrapper
- [x] Work on wrapper for alternating step-update procedure
- [x] Add a ER buffer style class for both on and off-policy RL
- [x] Naively replace step with catch env
- Do we also want to vmap over the update?! Learning rate has to be adapted in that case to be eta/batch_dim. E.g. make many small updates = 1 batch update
- This is mainly relevant for PG methods where the update is done outside anyways
- Also keep around state transition info - even if agent only sees obs
- Probably should rename `DeterministicRollouts` to `PlainRollouts` and make it work on stochastic policies.
- Test rollout wrappers on more policy types as well as environments
    - Write tests in the first place!
- Probably need a function to initialize the buffer. Fill it up until batchsize is met - to circumvent 'if' statement
    - E.g. run 1 episode with random policy
- Check if squeeze for obs/state is really required - different env?

## 26/01/21 - Start work on porting RLax DQN over
- [x] Refactor examples and notebooks
- [x] Run rlax DQN example + eval runtime = ca. 90 secs - 300 eps
- Considerations when replacing bsuite catch env with gymnax
    - Want to jit over entire episode rollout. So one wrapped loop
        for ep in range(train_eps):
            run_jitted_episode <- step, push, update
    - Rollout wrapper should look something like this
    ```python
    class DQNStyleWrapper:
        def __init__():
            ...
        def action_selection():
            ...
        def get_transition():
            ...
        def store_transition():
            ...
        def update_policy():
            ...
        def actor_learner_step():
            a = action_selection
            transition = get_transition(a)
            buffer = store_transition(transition)
            policy = update_policy(buffer)
        def lax_rollout():
            # scan over actor-learner-step
            ...
    ```
- [x] Rewrite base wrapper to be more abstract
- [x] Add trax example to episode wrappers = Problem with how inputs and params are passed to the policy - not params, input but other way around

## 25/01/21 - Clean Up animation for determ. policy
- [x] Move animator class from evosax to gymnax + example
- [x] Copy rlax DQN over from repo and run - get feeling for time

## 24/01/21 - Episode wrappers + bsuite catch
- [x] Get rollout wrapper to smoothly integrate with `evosax`
- [x] Refactor envs -> environments
- [x] Implement bsuite catch env
- [x] Move 2-arm Bernoulli bandit from evosax to gymnax
- Include timestep in state and terminate/set done?

## 19/01/21 - Start working on episode wrappers
- Want a template that wraps around `step`, `reset`, `make` and provides nice `lax.scan` if policy is given in the right format.
    - As much flexibility while giving highest level of abstraction
    - `policy_step` function is where the magic is happening
    - Allow for stochastic rollout
    - Flexible data storage - what are stats to store
- Base wrapper running as well for haiku, flax MLP policies
    - Do we also need a recurrent wrapper?
    - Implement PPO and DQN examples with rlax

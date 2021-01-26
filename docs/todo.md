## TODOs, Notes & Questions
- [ ] Add timeout condition with env_params and t in state!
- [ ] Episode rollout wrapper
    - [ ] Policy Gradient Wrapper with update only after transition
    - [ ] Q-Learning Wrapper with update after each transition
    - [ ] Deterministic vs. stochastic policy rollout
- [ ] Add TPU speed tests
- [ ] Add more environments
    - [ ] bsuite classic example
    - [ ] Gridworld/4 rooms
    - [ ] Toy text from gym
    - [ ] Simple bandit environments
- [ ] Add test for transition correctness compared to OpenAI gym
    - [x] Continuous Control
    - [x] Catch Bsuite
- [ ] Add backdoor for rendering in OpenAI gym
- [ ] Add random policy/sampling for basic rollout
- [ ] Figure out if numerical errors really matter

- [ ] Documentation
    - [ ] Pypi installation and github workflow for releases
    - [ ] Add state, obs, action space info as in dm_control paper
    - [ ] Connect notebooks with example Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk


## 19/01/21 - Start working on episode wrappers

- Want a template that wraps around `step`, `reset`, `make` and provides nice `lax.scan` if policy is given in the right format.
    - As much flexibility while giving highest level of abstraction
    - `policy_step` function is where the magic is happening
    - Allow for stochastic rollout
    - Flexible data storage - what are stats to store

- Base wrapper running as well for haiku, flax MLP policies
    - Do we also need a recurrent wrapper?
    - Implement PPO and DQN examples with rlax


## 24/01/21 - Episode wrappers + bsuite catch

- [x] Get rollout wrapper to smoothly integrate with `evosax`
- [x] Refactor envs -> environments
- [x] Implement bsuite catch env
- [x] Move 2-arm Bernoulli bandit from evosax to gymnax
- Include timestep in state and terminate/set done?


## 25/01/21 - Clean Up animation for determ. policy

- [x] Move animator class from evosax to gymnax + example
- [x] Copy rlax DQN over from repo and run - get feeling for time

## 26/01/21 - Start work on porting RLax DQN over

- [x] Refactor examples and notebooks
- [x] Run rlax DQN example + eval runtime = ca. 90 secs - 300 eps

- Considerations when replacing bsuite catch env with gymnax
    - Want to jit over entire episode rollout. So one wrapped loop
        for ep in range(train_eps):
            run_jitted_episode <- step, push, update
    - Probably need a function to initialize the buffer. Fill it up until batchsize is met - to circumvent 'if' statement
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
- [x] Add trax example to episode wrappers

## 26/01/21 - Work on stochastic wrapper, value-based wrapper, PG wrapper

- [ ] Add a ER buffer style class for both on and off-policy RL
- [ ] Make decision on discount vs done syntax
- [ ] Naively replace step with catch env
- [ ] Work on wrapper for alternating step-update procedure

## Next thing to do

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

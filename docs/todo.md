## TODOs, Notes & Questions
- [ ] Episode rollout wrapper
    - [ ] Policy Gradient Wrapper with update only after transition
    - [ ] Q-Learning Wrapper with update after each transition
    - [ ] Deterministic vs. stochastic policy rollout
- [ ] Learn more about setup in dm_env
- [ ] Add TPU speed tests
- [ ] Add more environments
    - [ ] bsuite classics - catch for transfer of catch rlax dqn example
    - [ ] Gridworld/4 rooms
    - [ ] Toy text from gym
- [ ] Add test for transition correctness compared to OpenAI gym
    - [x] Continuous Control
- [ ] Add backdoor for rendering in OpenAI gym
- [ ] Add random policy
- [ ] Figure out if numerical errors really matter
- [ ] Connect notebooks with example Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk


## 19/01/20 - Start working on episode wrappers

- Want a template that wraps around `step`, `reset`, `make` and provides nice `lax.scan` if policy is given in the right format.
    - As much flexibility while giving highest level of abstraction
    - `policy_step` function is where the magic is happening
    - Allow for stochastic rollout and flexible data storage

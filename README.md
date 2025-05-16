<h1 align="center">
  <a href="https://github.com/RobertTLange/gymnax/blob/main/docs/logo.png"><img src="https://github.com/RobertTLange/gymnax/blob/main/docs/logo.png?raw=true" width="215" /></a><br>
  <b>Reinforcement Learning Environments in JAX 🌍</b><br>
</h1>

<p align="center">
  <a href="https://pypi.python.org/pypi/gymnax"><img src="https://img.shields.io/pypi/pyversions/cax.svg?style=flat" /></a>
  <a href= "https://badge.fury.io/py/gymnax"><img src="https://badge.fury.io/py/gymnax.svg" /></a>
  <a href= "https://github.com/RobertTLange/gymnax/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
  <a href= "https://codecov.io/gh/RobertTLange/gymnax"><img src="https://codecov.io/gh/RobertTLange/gymnax/branch/main/graph/badge.svg?token=OKKPDRIQJR" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
</p>

Are you fed up with slow CPU-based RL environment processes? Do you want to leverage massive vectorization for high-throughput RL experiments? `gymnax` brings the power of `jit` and `vmap`/`pmap` to the classic gym API. It supports a range of different environments including [classic control](https://github.com/openai/gym/tree/master/gym/envs/classic_control), [bsuite](https://github.com/deepmind/bsuite), [MinAtar](https://github.com/kenjyoung/MinAtar/) and a collection of classic/meta RL tasks. `gymnax` allows explicit functional control of environment settings (random seed or hyperparameters), which enables accelerated & parallelized rollouts for different configurations (e.g. for meta RL). By executing both environment and policy on the accelerator, it facilitates the Anakin sub-architecture proposed in the Podracer paper [(Hessel et al., 2021)](https://arxiv.org/pdf/2104.06272.pdf) and highly distributed evolutionary optimization (using e.g. [`evosax`](https://github.com/RobertTLange/evosax)). We provide training & checkpoints for both PPO & ES in [`gymnax-blines`](https://github.com/RobertTLange/gymnax-blines). Get started here 👉 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/gymnax/blob/main/examples/getting_started.ipynb).

## Basic `gymnax` API Usage 🍲

```python
import jax
import gymnax

key = jax.random.key(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("Pendulum-v1")

# Reset the environment.
obs, state = env.reset(key_reset, env_params)

# Sample a random action.
action = env.action_space(env_params).sample(key_act)

# Perform the step transition.
n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
```

## Implemented Accelerated Environments 🏎️


| Environment Name | Reference | Source | 🤖 Ckpt (Return) | Secs/1M 🦶 <br /> A100 (2k 🌎)
| --- | --- | --- | --- | --- |
| [`Acrobot-v1`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/acrobot.py) | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540)  | [Click](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/Acrobot-v1) (R: -80) | 0.07 
| [`Pendulum-v1`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/pendulum.py) | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540)  | [Click](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)  | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/Pendulum-v1) (R: -130) | 0.07
| [`CartPole-v1`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/cartpole.py) | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540)  | [Click](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/CartPole-v1) (R: 500) | 0.05
| [`MountainCar-v0`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/mountain_car.py) | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) | [Click](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/MountainCar-v0) (R: -118) | 0.07
| [`MountainCarContinuous-v0`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/continuous_mountain_car.py) | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540)  | [Click](https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/MountainCarContinuous-v0) (R: 92) | 0.09
|  |  |  |  | 
| [`Asterix-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/asterix.py) | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) | [Click](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/asterix.py) | [PPO](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/Asterix-MinAtar) (R: 15) | 0.92
| [`Breakout-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/breakout.py) | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) | [Click](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py) | [PPO](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/Breakout-MinAtar) (R: 28) | 0.19
| [`Freeway-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/freeway.py) | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) | [Click](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py) | [PPO](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/Freeway-MinAtar) (R: 58) | 0.87
| [`SpaceInvaders-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/space_invaders.py) | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) | [Click](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/space_invaders.py) | [PPO](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/SpaceInvaders-MinAtar) (R: 131) | 0.33
|  |  |  |  | 
| [`Catch-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/catch.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/Catch-bsuite) (R: 1) | 0.15
| [`DeepSea-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/deep_sea.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/DeepSea-bsuite) (R: 0) | 0.22
| [`MemoryChain-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/memory_chain.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/memory_chain.py)  | [PPO, ES](https://github.com/RobertTLange/tree/main/gymnax-blines/agents/MemoryChain-bsuite) (R: 0.1) | 0.13
| [`UmbrellaChain-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/umbrella_chain.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/umbrella_chain.py)  | [PPO, ES](https://github.com/RobertTLange/tree/main/gymnax-blines/agents/UmbrellaChain-bsuite) (R: 1) | 0.08
| [`DiscountingChain-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/discounting_chain.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/discounting_chain.py)  | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/DiscountingChain-bsuite) (R: 1.1) | 0.06
| [`MNISTBandit-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/mnist.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/mnist.py)  | - | -
| [`SimpleBandit-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/bandit.py) | [Osband et al. (2019)](https://openreview.net/forum?id=rygf-kSYwH) | [Click](https://github.com/deepmind/bsuite/blob/master/bsuite/environments/bandit.py)  | - | -
|  |  |  |  | 
| [`FourRooms-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/rooms.py) | [Sutton et al. (1999)](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) | [Click](https://github.com/howardh/gym-fourrooms) | [PPO, ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/FourRooms-misc) (R: 1) | 0.07
| [`MetaMaze-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/meta_maze.py) | [Micconi et al. (2020)](https://arxiv.org/abs/2002.10585)  | [Click](https://github.com/uber-research/backpropamine/blob/master/simplemaze/maze.py) | [ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/MetaMaze-misc) (R: 32) | 0.09
| [`PointRobot-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/point_robot.py) | [Dorfman et al. (2021)](https://openreview.net/pdf?id=IBdEfhLveS) | [Click](https://github.com/Rondorf/BOReL/blob/main/environments/toy_navigation/point_robot.py) | [ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/PointRobot-misc) (R: 10) | 0.08
| [`BernoulliBandit-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/bernoulli_bandit.py) | [Wang et al. (2017)](https://arxiv.org/abs/1611.05763) | [Click](https://github.com/RobertTLange/minimal-meta-rl/blob/main/bandits/bandit_env.py) | [ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/BernoulliBandit-misc) (R: 90) | 0.08
| [`GaussianBandit-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/gaussian_bandit.py) | [Lange & Sprekeler (2022)](https://arxiv.org/abs/2010.04466) | [Click](https://github.com/RobertTLange/learning-not-to-learn) | [ES](https://github.com/RobertTLange/gymnax-blines/tree/main/agents/GaussianBandit-misc) (R: 0) | 0.07
| [`Reacher-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/reacher.py) | [Lenton et al. (2021)](https://github.com/unifyai/gym/) | [Click](https://github.com/unifyai/gym/blob/master/ivy_gym/reacher.py) | 
| [`Swimmer-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/swimmer.py) | [Lenton et al. (2021)](https://github.com/unifyai/gym/) | [Click](https://github.com/unifyai/gym/blob/master/ivy_gym/swimmer.py) |
| [`Pong-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/pong.py) | [Kirsch (2018)](https://github.com/BlackHC/batch_pong_poc) | [Click](https://github.com/BlackHC/batch_pong_poc/blob/master/src/vanilla_pong.py) |  

\* All displayed speeds are estimated for 1M step transitions (random policy) on a NVIDIA A100 GPU using `jit` compiled episode rollouts with 2000 environment workers. For more detailed speed comparisons on different accelerators (CPU, RTX 2080Ti) and MLP policies, please refer to the [`gymnax-blines`](https://github.com/RobertTLange/gymnax-blines) documentation.


## Installation ⏳

The latest `gymnax` release can directly be installed from PyPI:

```
pip install gymnax
```

If you want to get the most recent commit, please install directly from the repository:

```
pip install git+https://github.com/RobertTLange/gymnax.git@main
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Examples 📖
* 📓 [Environment API](examples/getting_started.ipynb) - Get started with the basic `gymnax` API.
* 📓 [Distributed Anakin Agent](examples/01_anakin.ipynb) - Train an Anakin [(Hessel et al., 2021)](https://arxiv.org/pdf/2104.06272.pdf) agent on `SpaceInvaders-MinAtar`.
* 📓 [ES with `gymnax`](examples/02_evolution.ipynb) - Meta-evolve an LSTM controller that controls 2 link pendula of different lengths.
* 📓 [Bandit A2C Meta-RL](examples/03_meta_a2c.ipynb) - Meta-learn an A2C LSTM that learns to explore/exploit in multi-arm bandit tasks.
* 📓 [Trained baselines](https://github.com/RobertTLange/gymnax-blines) - Check out the trained baseline agents (PPO/ES) in `gymnax-blines`.

## Key Selling Points 💵

- **Environment vectorization & acceleration**: Easy composition of JAX primitives (e.g. `jit`, `vmap`, `pmap`):

  ```python
  # Jit-accelerated step transition
  jit_step = jax.jit(env.step)

  # map (vmap/pmap) across random keys for batch rollouts
  reset_key = jax.vmap(env.reset, in_axes=(0, None))
  step_key = jax.vmap(env.step, in_axes=(0, 0, 0, None))

  # map (vmap/pmap) across env parameters (e.g. for meta-learning)
  reset_params = jax.vmap(env.reset, in_axes=(None, 0))
  step_params = jax.vmap(env.step, in_axes=(None, 0, 0, 0))
  ```
  For speed comparisons with standard vectorized NumPy environments check out [`gymnax-blines`](https://github.com/RobertTLange/gymnax-blines).

- **Scan through entire episode rollouts**: You can also `lax.scan` through entire `reset`, `step` episode loops for fast compilation:

  ```python
  def rollout(key_input, policy_params, env_params, steps_in_episode):
      """Rollout a jitted gymnax episode with lax.scan."""
      # Reset the environment
      key_reset, key_episode = jax.random.split(key_input)
      obs, state = env.reset(key_reset, env_params)

      def policy_step(state_input, tmp):
          """lax.scan compatible step transition in jax env."""
          obs, state, policy_params, key = state_input
          key, key_step, key_net = jax.random.split(key, 3)
          action = model.apply(policy_params, obs)
          next_obs, next_state, reward, done, _ = env.step(
              key_step, state, action, env_params
          )
          carry = [next_obs, next_state, policy_params, key]
          return carry, [obs, action, reward, next_obs, done]

      # Scan over episode step loop
      _, scan_out = jax.lax.scan(
          policy_step,
          [obs, state, policy_params, key_episode],
          (),
          steps_in_episode
      )
      # Return masked sum of rewards accumulated by agent in episode
      obs, action, reward, next_obs, done = scan_out
      return obs, action, reward, next_obs, done
  ```

- **Build-in visualization tools**: You can also smoothly generate GIF animations using the `Visualizer` tool, which covers all `classic_control`, `MinAtar` and most `misc` environments: 
  ```python
  from gymnax.visualize import Visualizer

  state_seq, reward_seq = [], []
  key, key_reset = jax.random.split(key)
  obs, env_state = env.reset(key_reset, env_params)
  while True:
      state_seq.append(env_state)
      key, key_act, key_step = jax.random.split(key, 3)
      action = env.action_space(env_params).sample(key_act)
      next_obs, next_env_state, reward, done, info = env.step(
          key_step, env_state, action, env_params
      )
      reward_seq.append(reward)
      if done:
          break
      else:
        obs = next_obs
        env_state = next_env_state
  
  cum_rewards = jnp.cumsum(jnp.array(reward_seq))
  vis = Visualizer(env, env_params, state_seq, cum_rewards)
  vis.animate(f"docs/anim.gif")
  ```

- **Training pipelines & pretrained agents**: Check out [`gymnax-blines`](https://github.com/RobertTLange/gymnax-blines) for trained agents, expert rollout visualizations and PPO/ES pipelines. The agents are minimally tuned, but can help you get up and running.

- **Simple batch agent evaluation**: *Work-in-progress*.
  ```python
  from gymnax.experimental import RolloutWrapper

  # Define rollout manager for pendulum env
  manager = RolloutWrapper(model.apply, env_name="Pendulum-v1")

  # Simple single episode rollout for policy
  obs, action, reward, next_obs, done, cum_ret = manager.single_rollout(key, policy_params)

  # Multiple rollouts for same network (different key, e.g. eval)
  key_batch = jax.random.split(key, 10)
  obs, action, reward, next_obs, done, cum_ret = manager.batch_rollout(
      key_batch, policy_params
  )

  # Multiple rollouts for different networks + key (e.g. for ES)
  batch_params = jax.tree.map(  # Stack parameters or use different
      lambda x: jnp.tile(x, (5, 1)).reshape(5, *x.shape), policy_params
  )
  obs, action, reward, next_obs, done, cum_ret = manager.population_rollout(
      key_batch, batch_params
  )
  ```

## Resources & Other Great Tools 📝
* 💻 [Brax](https://github.com/google/brax): JAX-based library for rigid body physics by Google Brain with JAX-style MuJoCo substitutes.
* 💻 [envpool](https://github.com/sail-sg/envpool): Vectorized parallel environment execution engine.
* 💻 [Jumanji](https://github.com/instadeepai/jumanji): A suite of diverse and challenging RL environments in JAX.
* 💻 [Pgx](https://github.com/sotetsuk/pgx): JAX-based classic board game environments.

### Acknowledgements & Citing `gymnax` ✏️

If you use `gymnax` in your research, please cite it as follows:

```
@software{gymnax2022github,
  author = {Robert Tjarko Lange},
  title = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url = {http://github.com/RobertTLange/gymnax},
  version = {0.0.4},
  year = {2022},
}
```

We acknowledge financial support by the [Google TRC](https://sites.research.google/trc/about/) and the Deutsche
Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2002/1 ["Science of Intelligence"](https://www.scienceofintelligence.de/) - project number 390523135.

## Development 👷

You can run the test suite via `python -m pytest -vv --all`. If you find a bug or are missing your favourite feature, feel free to create an issue and/or start [contributing](CONTRIBUTING.md) 🤗.

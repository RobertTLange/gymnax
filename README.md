# Classic Gym Environments in JAX 🏎️
[![Pyversions](https://img.shields.io/pypi/pyversions/gymnax.svg?style=flat-square)](https://pypi.python.org/pypi/gymnax)
[![PyPI version](https://badge.fury.io/py/gymnax.svg)](https://badge.fury.io/py/evosax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/RobertTLange/gymnax/branch/main/graph/badge.svg?token=OKKPDRIQJR)](https://codecov.io/gh/RobertTLange/gymnax)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/gymnax/blob/main/examples/getting_started.ipynb)
<a href="https://github.com/RobertTLange/gymnax/blob/main/docs/gymnax_logo.png?raw=true"><img src="https://github.com/RobertTLange/gymnax/blob/main/docs/gymnax_logo.png?raw=true" width="200" align="right" /></a>

Are you fed up with slow CPU-based RL environment processes? Do you want to leverage massive vectorization for high-throughput RL experiments? `gymnax` brings the power of `jit` and `vmap` to classic OpenAI gym environments.

## Basic `gymnax` API Usage 🍲

- Classic Open AI gym wrapper including `gymnax.make`, `env.reset`, `env.step`:

```python
import jax
import gymnax

rng = jax.random.PRNGKey(0)
rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

env, env_params = gymnax.make("Pendulum-v1")

obs, state = env.reset(key_reset, env_params)
action = env.action_space.sample(key_policy)
n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
```

## Episode Rollouts, Vectorization & Acceleration 🏍️
- Easy composition of JAX primitives (e.g. `jit`, `vmap`, `pmap`):

```python
def rollout(rng_input, policy_params, env_params, num_env_steps):
      """Rollout a jitted gymnax episode with lax.scan."""
      # Reset the environment
      rng_reset, rng_episode = jax.random.split(rng_input)
      obs, state = env.reset(rng_reset, env_params)

      def policy_step(state_input, tmp):
          """lax.scan compatible step transition in jax env."""
          obs, state, policy_params, rng = state_input
          rng, rng_step, rng_net = jax.random.split(rng, 3)
          action = network.apply({"params": policy_params}, obs, rng=rng_net)
          next_o, next_s, reward, done, _ = env.step(
              rng_step, state, action, env_params
          )
          carry = [next_o.squeeze(), next_s, policy_params, rng]
          return carry, [reward, done]

      # Scan over episode step loop
      _, scan_out = jax.lax.scan(
          policy_step,
          [obs, state, policy_params, rng_episode],
          [jnp.zeros((num_env_steps, 2))],
      )
      # Return masked sum of rewards accumulated by agent in episode
      rewards, dones = scan_out[0], scan_out[1]
      rewards = rewards.reshape(num_env_steps, 1)
      ep_mask = (jnp.cumsum(dones) < 1).reshape(num_env_steps, 1)
      return jnp.sum(rewards * ep_mask)
```

```python
# Jit-Compiled Episode Rollout
jit_rollout = jax.jit(rollout, static_argnums=3)

# Vmap across random keys for Batch Rollout
batch_rollout = jax.vmap(jit_rollout, in_axes=(0, None, None, None))
```

- Vectorization over different environment parametrizations:

```python
env.step(key_step, state, action, env_params)
```

## Implemented Accelerated Environments 🌍
<details><summary>
<a href="https://github.com/openai/gym/">Classic Control OpenAI gym</a> environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- |
| `Pendulum-v0` | ✅  | ✅ |
| `CartPole-v0` | ✅  | ✅ |
| `MountainCar-v0` | ✅  | ✅ |
| `MountainCarContinuous-v0` | ✅  | ✅ |
| `Acrobot-v1` | ✅  | ✅ |
</details>

<details><summary>
<a href="https://github.com/deepmind/bsuite/">DeepMind's BSuite</a> environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- |
| `Catch-bsuite` | ✅  | ✅ |
| `DeepSea-bsuite` | ✅  | ✅ |
| `MemoryChain-bsuite` | ✅  | ✅ |
| `UmbrellaChain-bsuite` | ✅  | ✅ |
| `DiscountingChain-bsuite` | ✅  | ✅ |
| `MNISTBandit-bsuite` | ✅  | ✅ |
| `SimpleBandit-bsuite` | ✅  | ✅ |
</details>

<details><summary>
<a href="https://github.com/kenjyoung/MinAtar">K. Young's and T. Tian's MinAtar</a> environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- |
| `Asterix-MinAtar` | ✅  | ✅ |
| `Breakout-MinAtar` | ✅  | ✅ |
| `Freeway-MinAtar` | ✅  | ✅ |
| `Seaquest-MinAtar` | ❌  | ❌ |
| `SpaceInvaders-MinAtar` | ✅  | ✅ |
</details>

<details><summary>
Miscellaneous Environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- |
| `BernoulliBandit-misc` | ✅  | ✅ |
| `GaussianBandit-misc` | ✅  | ✅ |
| `FourRooms-misc` | ✅  | ✅ |
</details>

## Installation 📝

`gymnax` can be directly installed from PyPi.

```
pip install gymnax
```

Alternatively, you can clone this repository and 'manually' install the `gymnax`:
```
git clone https://github.com/RobertTLange/gymnax.git
cd gymnax
pip install -e .
```

## Benchmarking Details 🚋

![](docs/classic_runtime_benchmark.png)

## Examples :school_satchel:
* :notebook: [Environment API](notebooks/getting_started.ipynb) - Check out the API and accelerated control environments.
* :notebook: [Anakin Agent](examples/getting_started.ipynb) - Check out the DeepMind's Anakin agent with `gymnax`'s `Catch-bsuite` environment.
* :notebook: [CMA-ES](examples/pendulum_cma_es.ipynb) - CMA-ES in JAX with vectorized population evaluation.

### Acknowledgements & Citing `gymnax` ✏️

To cite this repository:

```
@software{gymnax2021github,
  author = {Robert Tjarko Lange},
  title = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url = {http://github.com/RobertTLange/gymnax},
  version = {0.0.1},
  year = {2021},
}
```

Much of the design of `gymnax` has been inspired by the classic OpenAI gym RL environment API and DeepMind's JAX eco-system. I am grateful to the JAX team and Matteo Hessel for their support and motivating words. Finally, a big thank you goes out to the TRC team at Google for granting me TPU quota for benchmarking `gymnax`.

## Notes, Development & Questions ❓

- If you find a bug or want a new feature, feel free to contact me [@RobertTLange](https://twitter.com/RobertTLange) or create an issue :hugs:
- You can check out the history of release modifications in [`CHANGELOG.md`](CHANGELOG.md) (*added, changed, fixed*).
- You can find a set of open milestones in [`CONTRIBUTING.md`](CONTRIBUTING.md).

<details>
  <summary>Design Notes (control flow, random numbers, episode termination). </summary>

1. Each step transition requires you to pass a set of environment parameters `env.step(rng, state, action, env_params)`, which specify the  'hyperparameters' of the environment. You can
2. `gymnax` automatically resets an episode after termination. This way we can ensure that trajectory rollouts with fixed amounts of steps continue rolling out transitions.
3. If you want calculate evaluation returns simply mask the sum using the binary discount vector.
</details>

# `gymnax` - Bringing `jit` & `vmap` to OpenAI's gym

Are you fed up with slow CPU-based RL environment processes? Do you want to leverage massive vectorization for high-throughput RL experiments? This repo allows you to accelerate not only your model, but also exploit XLA and the power of JAX's primitives for a collection of classic Open AI gym environments.

## Basic API Usage

```python
import jax
import jax.numpy as jnp
from gymnax import make_env

rng = jax.random.PRNGKey(0)
rng, rng_input = jax.random.split(rng)

reset, step, env_params = make_env("Pendulum-v0")
obs, state = reset(rng)
action = your_jax_policy(model_params, obs)
next_obs, next_state, reward, done, _ = step(env_params, state, action)
```

## Benchmarks & Speed Gains

<details>
  <summary>Device and benchmark details.</summary>
The speed comparisons were benchmarked for the devices and transition rollout settings listed below. Multi-episode rollouts are collected synchronously and using a composition of `jit`, `vmap`/`pmap` (over episodes) and `lax.scan` (over the action-perception/RL loop).

| Name | Framework | Description | Device | Steps in Ep. | Number of Ep. |
|:---:|:---:|:---:| :---:| :---:| :---:| :---:|
CPU-STEP-GYM | OpenAI gym/NumPy | Single transition |2,7 GHz Intel Core i7| 1 | - |
CPU-STEP-JAX | `gymnax`/JAX | Single transition |2,7 GHz Intel Core i7| 1 | - |
CPU-RANDOM-GYM | OpenAI gym/NumPy | Random episode |2,7 GHz Intel Core i7| 200 | 1 |
CPU-RANDOM-JAX | `gymnax`/JAX | Random episode |2,7 GHz Intel Core i7| 200 | 1 |
CPU-FFW-64-GYM-TORCH | OpenAI gym/NumPy + PyTorch | 1-Hidden Layer MLP (64 Units) | 2,7 GHz Intel Core i7| 200 | 1 |
CPU-FFW-64-JAX | `gymnax`/JAX |  1-Hidden Layer MLP (64 Units) | 2,7 GHz Intel Core i7| 200 | 1 |
GPU-FFW-64-GYM-TORCH | OpenAI gym/NumPy + PyTorch | 1-Hidden Layer MLP (64 Units) | GeForce RTX 2080Ti | 200 | 1
GPU-FFW-64-JAX | `gymnax`/JAX |  1-Hidden Layer MLP (64 Units) | GeForce RTX 2080Ti | 200 | 1
TPU-FFW-64-JAX | `gymnax`/JAX | JAX 1-Hidden Layer MLP (64 Units) | GCP TPU VM | 200 | 1
GPU-FFW-64-JAX-2000 | `gymnax`/JAX | 1-Hidden Layer MLP (64 Units) | GeForce RTX 2080Ti | 200 | 2000
TPU-FFW-64-JAX-2000 | `gymnax`/JAX | 1-Hidden Layer MLP (64 Units) | GCP TPU VM | 200 | 2000

</details>

### Classic Control Tasks


| Environment | `Pendulum-v0` | `CartPole-v1` | `MountainCar-v0` | `MountainCarContinuous-v0` | `Acrobot-v1` |
|:---:|:---:|:---:| :---:| :---:| :---:|
CPU-STEP-GYM |  | |  |  |
CPU-STEP-JAX |  | |  |  |
CPU-RANDOM-GYM | | | |
CPU-RANDOM-JAX | | | | |
CPU-FFW-64-GYM-TORCH |  |
CPU-FFW-64-JAX |
GPU-FFW-64-GYM-TORCH |
GPU-FFW-64-JAX |
GPU-FFW-64-JAX-2000 |
TPU-FFW-64-JAX-2000 |


## Installing `gymnax` and dependencies

Directly install from PyPi.

```
pip install gymnax
```

Alternatively, you can clone this repository and afterwards 'manually' install the toolbox (preferably in a clean Python 3.6 environment):

```
git clone https://github.com/RobertTLange/gymnax.git
cd gymnax
pip install -e .
```

This will install all required dependencies. Please note that `gymnax` is only tested for Python 3.6.


## Examples, Notebooks & Colabs
* :notebook: [Classic Control](examples/classic_control.ipynb) - Checkout `Pendulum-v0` and other accelerated control tasks.


## TODOs, Notes, Development & Questions
- [ ] Add backdoor for rendering in OpenAI gym
- [ ] Add test for transition correctness compared to OpenAI gym
- [ ] Connect notebooks with Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk

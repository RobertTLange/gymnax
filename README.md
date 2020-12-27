# `gymnax` - Bringing `jit` & `vmap` to OpenAI gym

Are you fed up with slow CPU-based RL environment processes? Do you want to leverage massive vectorization for high-throughput RL experiments? This repo allows you to accelerate not only your model, but also exploit XLA and the power of JAX's primitives for a collection of classic Open AI gym environments.

## Basic API Usage

```python
import jax
import jax.numpy as jnp
from gymnax import make_env

rng = jax.random.PRNGKey(0)
rng, key_reset, key_step = jax.random.split(rng, 3)

reset, step, env_params = make_env("Pendulum-v0")
obs, state = reset(key_reset)
action = your_jax_policy(policy_params, obs)
next_obs, next_state, reward, done, _ = step(key_step, env_params,
                                             state, action)
```


<details>
  <summary><code>jit</code>-ting entire episode rollouts & vectorization via <code>vmap</code>.
  </summary>

```python
def policy_step(state_input, tmp):
    """ lax.scan compatible step transition in JAX env. """
    rng, obs, state, policy_params, env_params = state_input
    rng, rng_input = jax.random.split(rng)
    action = your_jax_policy(policy_params, obs)
    next_o, next_s, reward, done, _ = step(rng_input, env_params,
                                           state, action)
    carry, y = [rng, next_o.squeeze(), next_s.squeeze(),
                policy_params, env_params], [reward]
    return carry, y


def policy_rollout(rng_input, policy_params, env_params, num_steps):
    """ Rollout a pendulum episode with lax.scan. """
    obs, state = reset(rng_input)
    scan_out1, scan_out2 = jax.lax.scan(policy_step,
                                        [rng_input, obs, state, policy_params, env_params],
                                        [jnp.zeros(num_steps)])
    return scan_out1, jnp.array(scan_out2)


# vmap across random keys used to initialize an episode
network_rollouts = jit(vmap(policy_rollout, in_axes=(0, None, None, None),
                            out_axes=0), static_argnums=(3))

rng, rng_input = jax.random.split(rng)
rollout_keys = jax.random.split(rng, num_episodes)
traces, rewards = network_rollouts(rollout_keys, network_params,
                                   env_params, num_env_steps)
```

</details>

<details>
  <summary>Important design questions (Random numbers). </summary>

1. All random number/PRNGKey handling has to be done explicitly outside of the function calls.

</details>

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

This will install all required dependencies. Please note that `gymnax` is only tested for Python 3.6. You can run the test from the repo directory via `pytest`.

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



## Examples, Notebooks & Colabs
* :notebook: [Classic Control](examples/classic_control.ipynb) - Checkout `Pendulum-v0` and other accelerated control tasks.


## TODOs, Notes, Development & Questions
- [ ] Add backdoor for rendering in OpenAI gym
- [ ] Add test for transition correctness compared to OpenAI gym
    - [ ] Pendulum-v0
- [ ] Connect notebooks with Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk

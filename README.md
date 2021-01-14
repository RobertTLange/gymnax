![](docs/gymnax_logo.png)

Are you fed up with slow CPU-based RL environment processes? Do you want to leverage massive vectorization for high-throughput RL experiments? `gymnax` brings the power of `jit` and `vmap` to classic OpenAI gym environments.

## Basic API Usage

```python
import jax, gymnax

rng, reset, step, env_params = gymnax.make("Pendulum-v0")
rng, key_reset, key_step = jax.random.split(rng, 3)

obs, state = reset(key_reset, env_params)
action = your_jax_policy(policy_params, obs)
next_obs, next_state, reward, done, _ = step(key_step, env_params,
                                             state, action)
```

<details><summary>
Implemented classic OpenAI environments.

</summary>

| Environment Class | Environment Name | Implemented | Tested | Single Step Speed Gain (Estimate vs. OpenAI) |
| --- | --- | --- | --- | --- |
| Classic Control | `Pendulum-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| Classic Control | `CartPole-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| Classic Control | `MountainCar-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| Classic Control | `MountainCarContinuous-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| Classic Control | `Acrobot-v1` | :heavy_check_mark:  | :heavy_check_mark: |
</details>

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
    obs, state = reset(rng_input, env_params)
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
  <summary>Important design questions (control flow, random numbers, episode termination). </summary>

1. All random number/PRNGKey handling has to be done explicitly outside of the environment function calls. This allows for more control and less opacity.
2. Each step transition requires you to pass a set of environment parameters `step(., env_params, .)`, which specify the transition/reward function. We do not have to `jit` over this axis and hence you are flexible to incorporate environment non-stationarities of your choosing!
3. Episode termination has to be handled outside of the simple transition call. This could for example be done using a placeholder output in the scanned function.
4. The estimated speed gains may depend on your hardware as well as your specific policy parametrization. In general this will also depend on how much parallelism your algorithm utilizes and the episode length through which we `scan` + `jit`.
5. Boolean conditionals are eliminated by replacing them by weighted sums. E.g.: `r_effective = r * (1 - done) + r_term * done`

</details>

## Installing `gymnax` and dependencies

`gymnax` can be directly installed from PyPi.

```
pip install gymnax
```

Alternatively, you can clone this repository and afterwards 'manually' install the toolbox (preferably in a clean Python 3.6 environment):

```
git clone https://github.com/RobertTLange/gymnax.git
cd gymnax
pip install -e .
```

This will install all required dependencies. Note that by default the `gymnax` installation will install CPU-only `jaxlib`. In order to install the CUDA-supported version, simply upgrade to the right `jaxlib`. E.g. for a CUDA 10.1 driver:

```
pip install --upgrade jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

You can find more details in the [JAX documentation](https://github.com/google/jax#installation). Finally, please note that `gymnax` is only tested for Python 3.6. You can directly run the test from the repo directory via `pytest`.

## Benchmarking Details

<details> <summary>
  Device and benchmark details.

</summary>

| Name | Framework | Description | Device | Steps in Ep. | Number of Ep. |
| --- | --- | --- | --- | --- | --- |
CPU-STEP-GYM | OpenAI gym/NumPy | Single transition |2,7 GHz Intel Core i7| 1 | - |
CPU-STEP-JAX | gymnax/JAX | Single transition |2,7 GHz Intel Core i7| 1 | - |
CPU-STEP-GYM | OpenAI gym/NumPy | Single transition |2,7 GHz Intel Core i7| 1 | - |
CPU-STEP-JAX | gymnax/JAX | Single transition |2,7 GHz Intel Core i7| 1 | - |
CPU-RANDOM-GYM | OpenAI gym/NumPy | Random episode |2,7 GHz Intel Core i7| 200 | 1 |
CPU-RANDOM-JAX | gymnax/JAX | Random episode |2,7 GHz Intel Core i7| 200 | 1 |
CPU-FFW-64-GYM-TORCH | OpenAI gym/NumPy + PyTorch | 1-Hidden Layer MLP (64 Units) | 2,7 GHz Intel Core i7| 200 | 1 |
CPU-FFW-64-JAX | gymnax/JAX |  1-Hidden Layer MLP (64 Units) | 2,7 GHz Intel Core i7| 200 | 1 |
GPU-FFW-64-GYM-TORCH | OpenAI gym/NumPy + PyTorch | 1-Hidden Layer MLP (64 Units) | GeForce RTX 2080Ti | 200 | 1 |
GPU-FFW-64-JAX | gymnax/JAX |  1-Hidden Layer MLP (64 Units) | GeForce RTX 2080Ti | 200 | 1 |
TPU-FFW-64-JAX | gymnax/JAX | JAX 1-Hidden Layer MLP (64 Units) | GCP TPU VM | 200 | 1 |
GPU-FFW-64-JAX-2000 | gymnax/JAX | 1-Hidden Layer MLP (64 Units) | GeForce RTX 2080Ti | 200 | 2000 |
TPU-FFW-64-JAX-2000 | gymnax/JAX | 1-Hidden Layer MLP (64 Units) | GCP TPU VM | 200 | 2000 |
</details>


The speed comparisons were benchmarked for the devices and transition rollout settings listed above. Multi-episode rollouts are collected synchronously and using a composition of `jit`, `vmap`/`pmap` (over episodes) and `lax.scan` (over the action-perception/RL loop).

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


## Contributing and Development

You can find a template and instructions for how to add a new environment [here](templates/env_template). Feel free to ping me ([@RobertTLange](https://twitter.com/RobertTLange)), open an issue or start contributing yourself.

## TODOs, Notes & Questions
- [ ] Add different speed tests - Gym, Torch, CPU, GPU, TPU, etc.
- [ ] Add test for transition correctness compared to OpenAI gym
    - [x] Continuous Control
- [ ] Add state, observation, action space table description of envs
- [ ] Add backdoor for rendering in OpenAI gym
- [ ] Add some random action sampling utility ala env.action_space.sample()
- [ ] Figure out if numerical errors really matter
- [ ] Connect notebooks with example Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk

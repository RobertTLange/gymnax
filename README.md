# Gymnax
## JAX-Based Reinforcement Learning Gym API
[![Pyversions](https://img.shields.io/pypi/pyversions/mle-toolbox.svg?style=flat-square)](https://pypi.python.org/pypi/mle-toolbox)[![Docs Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/RobertTLange/mle-toolbox/) [![PyPI version](https://badge.fury.io/py/mle-toolbox.svg)](https://badge.fury.io/py/mle-toolbox)
<a href="docs/gymnax_logo.png"><img src="docs/gymnax_logo.png" width="200" align="right" /></a>

Are you fed up with slow CPU-based RL environment processes? Do you want to leverage massive vectorization for high-throughput RL experiments? `gymnax` brings the power of `jit` and `vmap` to classic OpenAI gym environments.

## Basic `gymnax` API Usage :stew:

- Classic Open AI gym wrapper including `gymnax.make`, `env.reset`, `env.step`:

```python
import jax, gymnax

rng, env = gymnax.make("Pendulum-v0")
rng, key_reset, key_step = jax.random.split(rng, 3)

obs, state = env.reset(key_reset)
action = your_jax_policy(policy_params, obs)
n_obs, n_state, reward, done, _ = env.step(key_step, state, action)
```

- Easy composition of JAX primitives (e.g. `jit`, `vmap`, `pmap`):

```python
jitted_step = jax.jit(env.step)
jitted_reset = jax.jit(env.reset)
```

- Vectorization over different environment parametrizations:

```python
env.step(key_step, state, action, env_params)
```

## Implemented Accelerated Environments :earth_africa:
<details><summary>
<a href="https://github.com/openai/gym/">Classic Control OpenAI gym</a> environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- | --- |
| `Pendulum-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| `CartPole-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| `MountainCar-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| `MountainCarContinuous-v0` | :heavy_check_mark:  | :heavy_check_mark: |
| `Acrobot-v1` | :heavy_check_mark:  | :heavy_check_mark: |
</details>

<details><summary>
<a href="https://github.com/deepmind/bsuite/">DeepMind's BSuite</a> environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- |
| `Catch-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
| `DeepSea-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
| `MemoryChain-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
| `UmbrellaChain-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
| `DiscountingChain-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
| `MNISTBandit-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
| `SimpleBandit-bsuite` | :heavy_check_mark:  | :heavy_check_mark: |
</details>

<details><summary>
<a href="https://github.com/kenjyoung/MinAtar">K. Young's and T. Tian's MinAtar</a> environments.

</summary>

| Environment Name | Implemented | Tested | Single Step Speed Gain (JAX vs. NumPy) |
| --- | --- | --- | --- |
| `Asterix-MinAtar` | :heavy_check_mark:  | :heavy_check_mark: |
| `Breakout-MinAtar` | :heavy_check_mark:  | :heavy_check_mark: |
| `Freeway-MinAtar` | :heavy_check_mark:  | :heavy_check_mark: |
| `Seaquest-MinAtar` | :x:  | :x: |
| `SpaceInvaders-MinAtar` | :heavy_check_mark:  | :heavy_check_mark: |
</details>

## Installation :memo:

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

Note that by default the `gymnax` installation will install CPU-only `jaxlib`. In order to install the CUDA-supported version, simply upgrade to the right `jaxlib`. E.g. for a CUDA 10.1 driver:

```
pip install --upgrade jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

You can find more details in the official [JAX documentation](https://github.com/google/jax#installation).

## Benchmarking Details :train:

![](docs/classic_runtime_benchmark.png)

## Examples :school_satchel:
* :notebook: [Environment API](notebooks/classic_control.ipynb) - Check out the API and accelerated control environments.
* :notebook: [Anakin Agent](examples/catch_anakin.ipynb) - Check out the DeepMind's Anakin agent with `gymnax`'s `Catch-bsuite` environment.
* :notebook: [CMA-ES](examples/catch_anakin.ipynb) - Check out the DeepMind's Anakin agent with `gymnax`'s `Catch-bsuite` environment.

### Acknowledgements & Citing `gymnax` :pencil2:

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

Much of the design of `gymnax` has been inspired by the classic OpenAI gym RL environment API. It relies on bits and pieces from DeepMind's JAX eco-system. I am grateful to the JAX team and Matteo Hessel for their support and motivating words. Finally, a big thank you goes out to the TRC team at Google for granting me TPU quota for benchmarking `gymnax`.

## Notes, Development & Questions :question:

- If you find a bug or want a new feature, feel free to contact me [@RobertTLange](https://twitter.com/RobertTLange) or create an issue :hugs:
- You can check out the history of release modifications in [`CHANGELOG.md`](CHANGELOG.md) (*added, changed, fixed*).
- You can find a set of open milestones in [`CONTRIBUTING.md`](CONTRIBUTING.md).
<details>
  <summary>Important design questions (control flow, random numbers, episode termination). </summary>

1. All random number/PRNGKey handling has to be done explicitly outside of the environment function calls. This allows for more control and less opacity.
2. Each step transition requires you to pass a set of environment parameters `step(., env_params, .)`, which specify the transition/reward function. We do not have to `jit` over this axis and hence you are flexible to incorporate environment non-stationarities of your choosing!
3. Episode termination has to be handled outside of the simple transition call. This could for example be done using a placeholder output in the scanned function.
4. The estimated speed gains may depend on your hardware as well as your specific policy parametrization. In general this will also depend on how much parallelism your algorithm utilizes and the episode length through which we `scan` + `jit`.
5. Boolean conditionals are eliminated by replacing them by weighted sums. E.g.: `r_effective = r * (1 - done) + r_term * done`

</details>

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
Wrapper!
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

![](docs/classic_runtime_benchmark.png)

## Examples, Notebooks & Colabs
* :notebook: [Classic Control](examples/classic_control.ipynb) - Checkout the API and accelerated control tasks.

## Citing `gymnax`

To cite this repository:

```
@software{gymnax2021github,
  author = {Robert Tjarko Lange},
  title = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url = {http://github.com/google/jax},
  version = {0.0.1},
  year = {2021},
}
```

## Contributing & Development

You can find a template and instructions for how to add a new environment [here](templates/env_template). Feel free to ping me ([@RobertTLange](https://twitter.com/RobertTLange)), open an issue or start contributing yourself.

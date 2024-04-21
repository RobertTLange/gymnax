### [v0.0.8] - 21/04/2024

Fix import errors for `matplotlib` and `seaborn` visualization.

### [v0.0.7] - 04/04/2024

1. Fixed most of the typing issues under pylint --strict.
1.1 This required unifying the interface of `step`, `step_env`, `get_obs`, 'is_terminal` for proper inheritance. 

2. `spaces` now depends on `gymnasium` instead of `gym`, adding an additional dependency. Maybe later on `gym` dependency can be completely removed since it's not maintained anymore? haven't looked into that.
3. Removed `_DeviceArray` from the tests as it's deprecated. Uses jax version `0.4.24`
4. Ran everything in `python 3.10` so technically you could support that too now.
5. In `wrapper/brax.py`, `GymnaxtoBraxWrapper` has two new methods `backend` and `observation_size`, as its required to be defined under newest brax version. Used brax version `0.10.0`.

Thanks to @Aidandos !


### [v0.0.6] - 12/04/2023

##### Added

- Gym, EvoJAX, Brax, DM env wrappers
- Reacher environment inspired by [Lenton et al. (2021)](https://github.com/unifyai/gym/)
- Swimmer environment inspired by [Lenton et al. (2021)](https://github.com/unifyai/gym/)
- Basic Pong environment inspired by [Kirsch (2018)](https://github.com/BlackHC/batch_pong_poc)

##### Fixed

- Fixed Minatar tests for jax arrays.
- Fixed reward setting in `DiscountingChain-v0`
- Fixed `Tuple` space check via enumerate.

##### Changed

- Refactored wrappers into separate sub-directory.

### [v0.0.5] - 24/08/2022
##### Fixed

- Fix deprecated `tree_multimap`.
- Fix device grabbing when using an `jnp.array` to set default in Acrobot env.

### [v0.0.3] - 15/06/2022
##### Fixed

- Fix import structure.

### [v0.0.2] - 15/06/2022

##### Added

- Fully functional API using flax-style `EnvState` and `EnvParams` for calling `env.step`. 
- MinAtar environments are not operational yet.
- Release to ensure that `evosax` 0.0.9 release can work with `GymFitness` backend in `gymnax`.

##### Changed

- Basically everything :)

### [v0.0.1] - 22/11/2021

##### Added
- Adds main `gym`-like infrastructure that is `jit`, `vmap` & `pmap` compatible:
    - Jittable environment base class: `environment.py`
    - Spaces: `Discrete`, `Continuous`, `Box`
    - Translation of numpy state to state dictionary for JAX `step`.
- Adds base set of environments:
    - OpenAI's `classic_control`: `Pendulum-v0`, `CartPole-v0`, `MountainCar-v0`, `ContinuousMountainCar-v0`, `Acrobot-v0`
    - DeepMind's `bsuite`: `Catch-bsuite`, `DeepSea-bsuite`, `DiscountingChain-bsuite`, `MemoryChain-bsuite`, `UmbrellaChain-bsuite`, `MNISTBandit-bsuite`, `SimpleBandit-bsuite`
    - `MinAtar`: `Asterix-MinAtar`, `Breakout-MinAtar`, `Freeway-MinAtar`,  `Seaquest-MinAtar`, `SpaceInvaders-MinAtar`
- Adds `tests` for comparing `gym`/`numpy` `reset` + `step`  with JAX version.
    - `tests/classic_control/test_gym_env.py`
    - `tests/bsuite/test_bsuite_env.py`
- Adds set of `notebooks` walking through the individual environments.
- Adds set of `examples` incorporating `gymnax` into JAX-based RL experiments.
    - Anakin agent - port DM Colab for gymnax
    - CMA-ES policy evolution - port blogpost with experimental minimal agent

##### Todo

- Adds benchmark infrastructure and numbers on different devices (CPU/GPU/TPU):
    - Transitions/Second & Specific rollout types vs Torch setup
        - CPU: Intel Xeon 2.4 GHz
        - GPU: V100, A100, RTX 2080Ti
        - TPU: V2, V3 - `vmap` + `pmap`
- Adds set of `experimental` utilities:
    - `dojos`: Multi-transition rollout wrapper via `lax.scan` + `jit` for sequential `step`.
    - `agents`: Minimal and evaluation agent wrappers.
- Adds miscellaneous environments: `Bandit-misc`, `Rooms-misc`

### [Unreleased]

- Added a concise Gymnax 1.0 migration summary to the README.
- Added a persistent `LogWrapper` validity signal that distinguishes reset
  placeholders from completed-episode returns and lengths.
- Added opt-in differentiable observations and states for the five built-in
  continuous-action environments while preserving stop-gradient defaults.
- Fixed DeepSea's default episode horizon to follow its grid size, avoiding
  oversized default rollouts.

### [v1.0.0] - 16/08/2026

- Added Gymnasium-compatible `FrozenLake-misc`, including configurable static
  maps, dynamics, rewards, and episode limits.
- Added public factory-based `gymnax.register` support for runtime custom
  environment registration.
- Added an external-environment path to `RolloutWrapper` for direct environment
  and complete-parameter injection.
- Added readable constructor-style representations for Gymnax space objects.
- Documented pixel-style observation tensors and their renderer distinction.
- Updated release, contribution, citation, and environment-space documentation.
- Modernized Apache-2.0 package metadata for current setuptools builds.
- Removed a public Codecov badge query token from repository documentation.

Thanks to @Dcyaprogrammer!
- **Breaking:** Released the Gymnax 1.0 six-value step API: `terminated` and
  `truncated` replace the legacy `done` result. Automatic reset and
  `info["final_observation"]` remain available for terminal transitions.
- Added PyTree-safe observation autoreset and the keyed `Environment.observe`
  extension point for new custom environments.
- Added opt-in `LegacyStepAPIWrapper` for applications that require the removed
  five-value step return while migrating.
- Updated Gymnasium, Brax, EvoJAX, DM-env, pure-RL, rollout, visualization, and
  tutorial integration paths for the 1.0 terminal contract.
- Published the accepted API and compatibility RFC: a non-breaking terminal
  metadata path for 0.x and the Gymnasium-style 1.0 migration contract.
- Added JIT/vmap-safe `terminated`, `truncated`, and `final_observation`
  metadata to the five-value environment API; Gymnasium adapters now forward
  distinct terminal and time-limit flags.
- Aligned DeepSea action mappings with canonical bsuite behavior, added the
  Seaquest MinAtar environment, and introduced an opt-in sticky-action wrapper.
- Restored Gymnasium classic-control GIF visualization with the modern
  `render_mode="rgb_array"` API and an optional `visualize` dependency extra.
- Fixed Acrobot autoreset state selection under JAX x64 mode.
- Fixed `LogWrapper` reset counter dtypes so wrapped environments work in JIT
  control flow.
- Fixed `Catch-bsuite` reset state dtypes for JAX x64 control flow.
- Fixed `MemoryChain-bsuite` reset and JIT stepping when `num_bits` is greater
  than one.
- Added a keyword-only `dtype` option to `spaces.Discrete`; it defaults to
  `int32` and supports `int64` when JAX x64 mode is enabled.
- Fixed `SimpleBandit` state initialization so `rewards`, `total_regret`, and `time` are assigned to the correct fields.
- Added a regression test covering `SimpleBandit` reset and step state construction.
- Fixed `RolloutWrapper` so `num_env_steps` overrides the default episode length during scans.
- Added a regression test covering non-default `num_env_steps` rollout shapes.

Thanks to @ponseko and @jinPrelude !

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

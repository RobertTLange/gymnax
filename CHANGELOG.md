### v0.0.1 - Unreleased

##### Added
- Adds main `gym`-like infrastructure that is `jit`, `vmap` & `pmap` compatible:
    - Jittable environment base class: `environment.py`
    - Spaces: `Discrete`, `Continuous`, `Box`
    - Translation of numpy state to state dictionary for JAX `step`.
- Adds base set of environments:
    - OpenAI's `classic_control`: `Pendulum-v0`, `CartPole-v0`, `MountainCar-v0`, `ContinuousMountainCar-v0`, `Acrobot-v0`
    - DeepMind's `bsuite`:
    - `MinAtar`:
    - Miscellaneous:
- Adds set of `tests` for environments comparing gym/`numpy` `reset` and `step` transitions with JAX-based version.
- Adds set of `notebooks` walking through the individual environments.

##### TODO
- Adds set of `examples` showing how to incorporate `gymnax` into JAX-based experiment pipelines.
    - Anakin agent
    - CMA-ES policy evolution
- Adds a set of benchmarks on different devices (CPU, GPU, TPU): Transitions/Sec vs Torch setup
    - CPU: Intel Xeon 2.4 GHz
    - GPU: V100, A100, RTX 2080Ti
    - TPU: V2, V3 - `vmap` + `pmap`
- Adds set of `experimental` utilities:
    - `dojos`: Multi-transition rollout via `lax.scan` + `jit` step transitions.
    - `agents`: Minimal and evaluation agent wrappers.

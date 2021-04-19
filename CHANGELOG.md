### v0.0.1 - Unreleased

##### Added
- Adds main infrastructure:
    - Jittable environment base class: `environment.py`
    - Spaces: `Discrete`, `Continuous`, `Box`
    - Translation of numpy state to state dictionary for JAX `step`.
- Adds base set of environments:
    - OpenAI's `classic_control`: `Pendulum-v0`, `CartPole-v0`, `MountainCar-v0`, `ContinuousMountainCar-v0`, `Acrobot-v0`
    - DeepMind's `bsuite`:
    - `MinAtar`:
    - Miscellaneous: 
- Adds set of `tests` for environments comparing gym/numpy `reset` and `step` transitions with JAX-based version.
- Adds set of `notebooks` walking through the individual environments.

##### TODO
- Adds set of `examples` showing how to incorporate `gymnax` into JAX-based experiment pipelines.
- Adds a set of benchmarks on different devices (CPU, GPU, TPU)
- Adds set of `experimental` utilities:
    - `dojos`: Multi-transition rollout via `lax.scan` + `jit` step transitions.
    - `agents`: Minimal and evaluation agent wrappers.

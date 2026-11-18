# Examples

The notebooks in
[`examples/`](https://github.com/RobertTLange/gymnax/tree/main/examples) are complete
programs that run in Colab.

| Notebook | Content | |
| --- | --- | --- |
| [Environment API](https://github.com/RobertTLange/gymnax/blob/main/examples/00_getting_started.ipynb) | `reset`, `step`, vectorization and compiled rollouts. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/gymnax/blob/main/examples/00_getting_started.ipynb) |
| [Distributed Anakin agent](https://github.com/RobertTLange/gymnax/blob/main/examples/01_anakin.ipynb) | An Anakin agent ([Hessel et al., 2021](https://arxiv.org/abs/2104.06272)) with environment and learner on the accelerator. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/gymnax/blob/main/examples/01_anakin.ipynb) |
| [Evolution strategies](https://github.com/RobertTLange/gymnax/blob/main/examples/02_evolution.ipynb) | Evolving pendulum controllers with [evosax](https://github.com/RobertTLange/evosax). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/gymnax/blob/main/examples/02_evolution.ipynb) |
| [Bandit meta-RL](https://github.com/RobertTLange/gymnax/blob/main/examples/03_meta_a2c.ipynb) | Meta-learning an A2C LSTM agent that explores and exploits in bandit tasks. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/gymnax/blob/main/examples/03_meta_a2c.ipynb) |

## Trained baselines

[gymnax-blines](https://github.com/RobertTLange/gymnax-blines) contains PPO and
evolution strategy training pipelines, trained checkpoints, rollout visualizations and
speed benchmarks for the gymnax environments.

## Related libraries

- [Brax](https://github.com/google/brax): rigid body physics in JAX.
- [Jumanji](https://github.com/instadeepai/jumanji): a diverse suite of JAX
  environments.
- [Pgx](https://github.com/sotetsuk/pgx): board games in JAX.
- [EnvPool](https://github.com/sail-sg/envpool): a vectorized execution engine for
  conventional environments.

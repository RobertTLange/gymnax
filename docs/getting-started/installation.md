# Installation

gymnax is available on PyPI:

```bash
pip install gymnax
```

To install the latest commit instead:

```bash
pip install git+https://github.com/RobertTLange/gymnax.git@main
```

This installs the CPU version of JAX. To run on a GPU or TPU, install the matching JAX
build first by following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Supported versions

| Dependency | Supported |
| --- | --- |
| Python | 3.10 to 3.14 |
| JAX | 0.6 to 0.11 (0.8.1 or newer on Python 3.14) |
| Gymnasium | 1.1 to 1.3 |

The bounds are declared with environment markers, so an unsupported combination fails
when resolving dependencies and not at import time.

## Optional extras

| Extra | Adds |
| --- | --- |
| `gymnax[visualize]` | Gymnasium's classic control renderer, needed to [animate](../guides/visualization.md) the classic control environments. |

```bash
pip install "gymnax[visualize]"
```

The Brax and EvoJAX [adapters](../guides/wrappers.md#adapters-to-other-libraries)
require `brax` and `evojax`, which you install yourself.

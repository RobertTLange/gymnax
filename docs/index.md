# gymnax

Reinforcement learning environments in JAX.

gymnax reimplements classic control tasks, bsuite, MinAtar and a set of meta-learning
environments as pure JAX functions behind the familiar `reset`/`step` API. State and
parameters are explicit, so every environment works with `jax.jit`, `jax.vmap` and
`jax.lax.scan`, and thousands of episodes run in parallel on a single accelerator
together with the policy.

```bash
pip install gymnax
```

## A first step

```python
import jax
import gymnax

key = jax.random.key(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# An environment and its default parameters
env, params = gymnax.make("Pendulum-v1")

obs, state = env.reset(key_reset, params)
action = env.action_space(params).sample(key_act)
obs, state, reward, terminated, truncated, info = env.step(
    key_step, state, action, params
)
```

Environments reset automatically when an episode ends. The last observation of the
finished episode is kept in `info["final_observation"]`.

## Where to go next

- [Quickstart](getting-started/quickstart.md) runs thousands of environments in
  parallel and compiles a full rollout with `jax.lax.scan`.
- [Concepts](guides/concepts.md) explains state, parameters, automatic resets and the
  difference between termination and truncation.
- [Environments](environments.md) lists every environment with its spaces.
- [API reference](reference/environment.md) documents all classes and functions.

## Citing gymnax

If you use gymnax in your research, please cite it:

```bibtex
@software{gymnax2022github,
    author  = {Robert Tjarko Lange},
    title   = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
    url     = {http://github.com/RobertTLange/gymnax},
    version = {1.1.0},
    year    = {2026},
}
```

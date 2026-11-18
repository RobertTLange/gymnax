# Concepts

## Environments are stateless

A gymnax environment is a collection of pure functions. Three kinds of data are passed
in explicitly:

| Object | Holds | Changes |
| --- | --- | --- |
| `key` | Randomness for one call | Every call |
| `state` | Everything that evolves during an episode | Every step |
| `params` | Settings of the dynamics, such as masses or the time limit | When you decide |

```python
import jax
import gymnax

env, params = gymnax.make("Pendulum-v1")
key_reset, key_step = jax.random.split(jax.random.key(0))

obs, state = env.reset(key_reset, params)
obs, state, reward, terminated, truncated, info = env.step(
    key_step, state, env.action_space(params).sample(key_step), params
)
```

State and parameters are immutable dataclasses and JAX PyTrees. Derive modified copies
with `replace`:

```python
params = params.replace(max_torque=1.0)
```

Since nothing is hidden inside the environment, `jax.vmap` can batch over keys, states
and parameters alike, and `jax.jit` compiles environment and policy into one program.

Some settings change array shapes or the structure of the computation, for example the
size of a grid. These are constructor arguments and are passed through `make`:

```python
catch_env, catch_params = gymnax.make("Catch-bsuite", rows=20, columns=10)
```

## Termination and truncation

`step` returns two flags, following Gymnasium:

- `terminated`: the episode reached a natural end, such as a fallen pole. The value of
  the following state is zero.
- `truncated`: the episode hit the time limit `params.max_steps_in_episode`. The task
  itself would have continued, so the following state still has value.

Both can be true in the same step. The episode is over when either one is.

## Automatic resets

When an episode ends, `step` resets the environment within the same call. The returned
`obs` and `state` already belong to the next episode, which keeps rollouts free of
Python control flow.

The observation that closed the finished episode is not lost. `info` always contains:

| Key | Meaning |
| --- | --- |
| `terminated` | Same as the returned flag. |
| `truncated` | Same as the returned flag. |
| `final_observation` | Observation of the transition before any reset. Equal to `obs` while the episode continues. |

Use these for value bootstrapping:

```python
import jax.numpy as jnp

obs, state, reward, terminated, truncated, info = env.step(
    key_step, state, env.action_space(params).sample(key_step), params
)

bootstrap_obs = info["final_observation"]
bootstrap_mask = 1.0 - terminated.astype(jnp.float32)
# target = reward + discount * bootstrap_mask * value(bootstrap_obs)
```

Only termination zeroes the bootstrap value. On truncation, bootstrap from
`final_observation` as usual.

To get the raw transition without the reset or the time limit, call `env.step_env`
with the same arguments.

## Spaces

`env.action_space(params)`, `env.observation_space(params)` and
`env.state_space(params)` return [spaces](../reference/spaces.md) that depend on the
parameters. They are jittable and offer `sample(key)` and `contains(x)`:

```python
action_space = env.action_space(params)
action = action_space.sample(jax.random.key(1))
assert action_space.contains(action)
```

Observations can be arrays or any PyTree of arrays; `reset`, `step` and the automatic
reset preserve the structure.

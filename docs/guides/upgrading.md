# Upgrading to 1.0

gymnax 1.0 changed the return value of `step` to match Gymnasium.

## What changed

=== "1.0"

    ```python
    obs, state, reward, terminated, truncated, info = env.step(
        key, state, action, params
    )
    done = terminated | truncated
    ```

=== "0.x"

    ```python
    obs, state, reward, done, info = env.step(key, state, action, params)
    ```

- `done` is split into `terminated` (natural end) and `truncated` (time limit).
- `info` always contains `terminated`, `truncated` and `final_observation`, the
  observation before the automatic reset.
- Custom environments implement `observe(key, state, action, params)`, and `step_env`
  returns `(obs, state, reward, terminated, info)` without handling the time limit.

## Migrating learning code

Replace `done` by `terminated | truncated` for episode bookkeeping. For value targets,
bootstrap from `info["final_observation"]` and mask only with `terminated`:

```python
bootstrap_obs = info["final_observation"]
bootstrap_mask = 1.0 - terminated.astype(jnp.float32)
```

See [Concepts](concepts.md#automatic-resets) for the reasoning.

## Keeping the old return value

Code that cannot migrate yet can wrap the environment:

```python
from gymnax.wrappers import LegacyStepAPIWrapper

legacy_env = LegacyStepAPIWrapper(env)
obs, state, reward, done, info = legacy_env.step(key, state, action, params)
```

The separate flags and `final_observation` remain available in `info`.

The [API RFC](../api_rfc.md) records the full contract and the reasoning behind it.

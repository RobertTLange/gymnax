# API and compatibility RFC

## Status

Accepted 2026-08-15. This RFC defines the next Gymnax API transition. It does
not change the current runtime API; the implementation work is tracked for the
next 0.x compatibility release and 1.0.

## Goals

- Preserve the JAX-friendly automatic-reset workflow while making terminal
  transitions safe for value bootstrapping.
- Align terminal semantics with Gymnasium: termination and time-limit
  truncation are separate signals.
- Make structured JAX PyTrees valid observations.
- Give custom environments one consistent, keyed and parameter-aware base
  contract.

## Current 0.x contract

The existing public API remains unchanged:

```python
observation, state, reward, done, info = env.step(key, state, action, params)
```

`done` means `terminated | truncated`. Gymnax continues to automatically reset
when `done` is true: returned `observation` and `state` begin the next episode.

The next compatible 0.x release will always add these keys to `info`:

| Key | Meaning |
| --- | --- |
| `terminated` | Natural end of an episode, excluding the time limit. |
| `truncated` | The transition reached `params.max_steps_in_episode`. |
| `final_observation` | The observation before automatic reset. |

The keys are JAX arrays or PyTrees with a stable structure for every transition,
so they work under `jax.jit` and `jax.vmap`. On nonterminal transitions,
`final_observation` equals the returned observation and callers must ignore it
unless `done` is true. If natural termination and the time limit occur together,
both flags are true. Bootstrap targets use `final_observation`; a natural
termination has zero continuation value even when `truncated` is also true.

`final_observation` comes directly from the environment-specific transition,
before the base environment selects a fresh reset observation. Gymnax will not
expose a terminal environment state through `info` in 0.x.

## 1.0 contract

Version 1.0 replaces the fourth `done` value with separate flags:

```python
observation, state, reward, terminated, truncated, info = env.step(
    key, state, action, params
)
```

Automatic reset remains the standard contract. `observation` and `state` still
refer to the next episode after either flag is true; `info["final_observation"]`
is the transition observation needed for learning from the completed episode.

`observation` may be any JAX PyTree whose leaves are arrays. `reset`, `step`,
autoreset, `jax.jit`, and `jax.vmap` preserve that tree structure. An
environment's `observation_space(params)` must describe the same tree.

### Environment implementation hooks

1. `step_env(key, state, action, params)` returns the raw transition
   `(observation, next_state, reward, terminated, info)`. It is responsible for
   natural termination, including action-dependent terminal conditions.
2. The base `Environment.step` computes `truncated` from the post-transition
   `next_state.time` and `params.max_steps_in_episode`; time limits must not be
   folded into `terminated`.
3. `observe(key, state, action, params)` replaces the inconsistent `get_obs`
   overloads. `action` is `None` for reset observations and the action that
   produced a transition otherwise. It may return a PyTree and supports noisy
   observations through `key`.
4. `is_terminal` and `discount` are no longer required base hooks. Environments
   that retain helpers must use them only for natural termination; learning code
   consumes the public terminal flags.

This permits POMDP observations, action-dependent ends, and parameter-dependent
or stochastic observations without custom signatures.

## Migration

For 0.x callers, preserve the five-value unpacking and use the new fields when
computing targets:

```python
next_obs, next_state, reward, done, info = env.step(key, state, action, params)
bootstrap_obs = info["final_observation"]
bootstrap_mask = 1.0 - info["terminated"].astype(jnp.float32)
```

For 1.0, replace `done` with `terminated, truncated`; reset control flow when
`terminated | truncated` is true. Retain `final_observation` for autoreset
rollouts. Gymnasium and vector-Gymnasium adapters must forward the two flags
without deriving one from the other.

## Delivery sequence

1. Implement the 0.x `info` fields with regression tests for ordinary,
   terminated, truncated, and simultaneous boundaries under JIT and vmap.
2. Audit every environment so `max_steps_in_episode` is only a truncation;
   update Gymnasium, Brax, rollout, logging, and evaluation wrappers.
3. Add PyTree observation coverage and migrate the base hooks.
4. Release 1.0 with the six-value API, deprecation/migration notes, and adapter
   contract tests.

Open issues #109, #38, #103, #107, #88, #59, #32, and #26 remain the public
discussion and implementation trackers. Pull request #108 is a useful
non-breaking reference for step 1, but requires a focused rebase after this RFC
and does not by itself resolve the API migration.

## Deferred decisions

This RFC intentionally does not consolidate constructor arguments into
`EnvParams`, expose terminal states, or define environment reward derivatives.
Those remain separate proposals after the terminal and observation contracts are
stable.

# API and compatibility RFC

## Status

Implemented for 1.0.0 on 2026-08-15. The 0.x terminal-metadata compatibility
release remains available in `805354b` and `a033a3e` for migration reference.

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

The current compatible 0.x API always adds these keys to `info`:

| Key | Meaning |
| --- | --- |
| `terminated` | Natural end of an episode, excluding the time limit. |
| `truncated` | The transition reached `params.max_steps_in_episode`. |
| `final_observation` | The observation before automatic reset. |

The keys have a stable JAX-array structure for every transition, so they work
under `jax.jit` and `jax.vmap`. On nonterminal transitions,
`final_observation` equals the returned observation and callers must ignore it
unless `done` is true. If natural termination and the time limit occur together,
both flags are true. Bootstrap targets use `final_observation`; a natural
termination has zero continuation value even when `truncated` is also true.

`final_observation` comes directly from the environment-specific transition,
before the base environment selects a fresh reset observation. Gymnax will not
expose a terminal environment state through `info` in 0.x.

Existing third-party environments that only override `is_terminal` retain their
previous reset behavior. To report a natural terminal transition that coincides
with a time limit, they must also override `is_terminated`; the legacy hook
cannot distinguish both causes by itself.

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
   observations through `key`. Built-in 1.0 environments continue to use a
   compatibility bridge to their existing `get_obs` implementations; new custom
   environments should implement `observe` directly.
4. `is_terminal` and `discount` are no longer required base hooks. Environments
   that retain helpers must use them only for natural termination; learning code
   consumes the public terminal flags.

This permits POMDP observations, action-dependent ends, and parameter-dependent
or stochastic observations without custom signatures.

## Migration

For code that must retain the historical five-value unpacking during migration,
wrap the 1.0 environment and use the retained metadata when computing targets:

```python
legacy_env = LegacyStepAPIWrapper(env)
next_obs, next_state, reward, done, info = legacy_env.step(key, state, action, params)
bootstrap_obs = info["final_observation"]
bootstrap_mask = 1.0 - info["terminated"].astype(jnp.float32)
```

For 1.0, replace `done` with `terminated, truncated`; reset control flow when
`terminated | truncated` is true. Retain `final_observation` for autoreset
rollouts. Gymnasium and vector-Gymnasium adapters forward the two flags without
deriving one from the other.

Code that cannot migrate immediately may use the opt-in
`gymnax.wrappers.LegacyStepAPIWrapper`. It restores the removed five-value
return and computes `done = terminated | truncated`; terminal metadata remains
available in `info`. New code and custom environments must implement the
six-value/`observe` contract directly. The legacy observation and terminal
helpers remain as compatibility internals for built-in implementations; they
are not required hooks for new environments.

## Delivery sequence

1. Completed: the 0.x `info` fields, built-in time-limit audit, and Gymnasium
   adapter forwarding have regression coverage for ordinary, terminated,
   truncated, simultaneous, JIT, and vmap transitions.
2. Completed: PyTree-safe autoreset, the keyed `observe` entrypoint, the
   six-value API, and the opt-in five-value compatibility adapter.
3. Released: 1.0.0 migration notes and wrapper contract coverage.

Issues #109, #38, #103, #107, #88, #59, #32, and #26 served as the public
discussion and implementation trackers. Pull request #108 was useful reference
material for the completed 0.x change, but did not itself resolve the 1.0 API
migration.

## Deferred decisions

The 1.0 RFC intentionally did not consolidate constructor arguments into
`EnvParams`, expose terminal states, or define transition derivatives. Those
questions were evaluated separately after the terminal and observation contracts
stabilized.

## Post-1.0 differentiable transition contract

Built-in environments retain their stop-gradient behavior by default. The five
continuous-action environments (`Pendulum-v1`, `MountainCarContinuous-v0`,
`PointRobot-misc`, `Reacher-misc`, and `Swimmer-misc`) declare support for
`env.with_transition_gradients()`. This method returns a new configured
environment; the original instance remains unchanged, and the new identity keeps
JIT caching explicit.

Configured environments preserve JAX gradients through transition observations
and floating-point state leaves. They do not return derivative values directly;
callers use `jax.grad`, `jax.jacrev`, or related transformations. Reward gradients
are unchanged by this mode. Unsupported environments reject the opt-in rather
than implying differentiability for discrete actions or event-driven dynamics.

The contract is pathwise and local to the executed JAX program with a fixed
random key. It does not promise smooth derivatives through clipping boundaries,
comparisons, discrete events, termination decisions, or PointRobot's conditional
respawn. Integer state leaves such as `time` are bookkeeping, not differentiation
targets.

Automatic reset retains the 1.0 semantics. After termination or truncation, the
publicly returned observation and state come from reset and have no action
gradient from the completed transition. `info["final_observation"]` preserves the
transition-observation gradient, while `step_env` exposes the raw next state
without autoreset.

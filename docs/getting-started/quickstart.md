# Quickstart

This page walks through a single transition, then batches it over many environments
and finally compiles a whole rollout.

## One transition

`gymnax.make` returns an environment and its default parameters. The environment itself
holds no episode data: `reset` returns a state, and `step` takes a state and returns
the next one.

```python
import jax
import jax.numpy as jnp
import gymnax

env, params = gymnax.make("CartPole-v1")

key = jax.random.key(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

obs, state = env.reset(key_reset, params)
action = env.action_space(params).sample(key_act)
obs, state, reward, terminated, truncated, info = env.step(
    key_step, state, action, params
)
```

Every source of randomness is an explicit PRNG key. The same key, state, action and
parameters always produce the same transition.

To change the dynamics, replace fields of the parameters:

```python
params = params.replace(max_steps_in_episode=200, gravity=5.0)
```

The [environment overview](../environments.md) links to the parameters of every
environment.

## Many environments at once

`reset` and `step` are ordinary JAX functions, so `jax.vmap` turns them into batched
versions. Here 2048 environments share one set of parameters:

```python
num_envs = 2048
reset = jax.vmap(env.reset, in_axes=(0, None))
step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

key, key_reset, key_step = jax.random.split(key, 3)
obs, state = reset(jax.random.split(key_reset, num_envs), params)

actions = jnp.zeros(num_envs, dtype=jnp.int32)
obs, state, reward, terminated, truncated, info = step(
    jax.random.split(key_step, num_envs), state, actions, params
)
print(obs.shape)  # (2048, 4)
```

Mapping over the parameters instead (`in_axes=(None, 0)` for `reset`) runs a different
variant of the environment in every slot, for example for meta-learning.

## A compiled rollout

Because environments reset automatically, a rollout is a fixed-length loop with no
Python control flow. `jax.lax.scan` compiles it into a single program:

```python
num_steps = 500


def rollout(key, policy_params):
    key_reset, key_rollout = jax.random.split(key)
    obs, state = env.reset(key_reset, params)

    def one_step(carry, key_step):
        obs, state = carry
        key_act, key_env = jax.random.split(key_step)
        action = policy(policy_params, obs, key_act)
        next_obs, state, reward, terminated, truncated, info = env.step(
            key_env, state, action, params
        )
        transition = (obs, action, reward, terminated, truncated)
        return (next_obs, state), transition

    keys = jax.random.split(key_rollout, num_steps)
    _, transitions = jax.lax.scan(one_step, (obs, state), keys)
    return transitions


def policy(policy_params, obs, key):
    """A random policy; replace it with your network."""
    return env.action_space(params).sample(key)


# 256 rollouts of 500 steps each, compiled once
batched_rollout = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
keys = jax.random.split(jax.random.key(1), 256)
obs, action, reward, terminated, truncated = batched_rollout(keys, None)
print(reward.shape)  # (256, 500)
```

`RolloutWrapper` in [`gymnax.experimental`](../reference/experimental.md) packages
this pattern.

## Next steps

- [Concepts](../guides/concepts.md) covers automatic resets and how to bootstrap
  correctly at the end of an episode.
- [Wrappers](../guides/wrappers.md) log episode returns or expose an environment
  through the Gymnasium API.
- [Custom environments](../guides/custom-environments.md) shows how to write your own.

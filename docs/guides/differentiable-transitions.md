# Differentiable transitions

By default, `step` stops gradients at the returned observation and state. Environments
with continuous actions can opt into differentiable transitions, for example for
model-based control or analytic policy gradients.

```python
import jax
import jax.numpy as jnp
import gymnax

env, params = gymnax.make("Pendulum-v1")
env = env.with_transition_gradients()

key_reset, key_step = jax.random.split(jax.random.key(0))
obs, state = env.reset(key_reset, params)


def next_angular_velocity(action):
    next_state = env.step(key_step, state, action, params)[1]
    return next_state.theta_dot


gradient = jax.grad(next_angular_velocity)(jnp.array([0.5]))
```

`with_transition_gradients()` returns a configured copy and leaves the original
environment unchanged. Call it before applying wrappers.

## Supported environments

| Environment | |
| --- | --- |
| `Pendulum-v1` | [reference](../reference/environments/classic_control/pendulum.md) |
| `MountainCarContinuous-v0` | [reference](../reference/environments/classic_control/continuous_mountain_car.md) |
| `PointRobot-misc` | [reference](../reference/environments/misc/point_robot.md) |
| `Reacher-misc` | [reference](../reference/environments/misc/reacher.md) |
| `Swimmer-misc` | [reference](../reference/environments/misc/swimmer.md) |

All other environments raise a `ValueError`. Check
`env.supports_transition_gradients` to find out in advance.

## What the gradients mean

- They are pathwise derivatives of the executed program for a fixed PRNG key.
- Clipping, comparisons, termination decisions and other discrete events are not
  smooth. The goal-triggered respawn of `PointRobot-misc` is discontinuous as well.
- Differentiate floating-point leaves of the state, such as `theta_dot` above. The
  state also holds integer bookkeeping such as `time`.
- Reward gradients do not depend on this setting. They are available wherever the
  reward function is differentiable.

## Interaction with automatic resets

When a step ends an episode, the returned observation and state come from the reset
and carry no gradient of the finished transition. The gradient remains available
through `info["final_observation"]`. Call `env.step_env` to get the raw next state
without the automatic reset.

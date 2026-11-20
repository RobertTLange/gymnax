# Custom environments

A custom environment subclasses `Environment` and implements a handful of hooks. The
base class supplies `reset` and `step`, including the time limit, the automatic reset
and the entries of `info`.

## A minimal example

The agent walks along a line and is rewarded for reaching a goal.

```python
import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    position: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    goal: float = 1.0
    step_size: float = 0.1
    max_steps_in_episode: int = 50


class Corridor(environment.Environment[EnvState, EnvParams]):
    """Walk along a line until reaching the goal."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(self, key, params):
        state = EnvState(position=jnp.array(0.0), time=0)
        return self.observe(key, state, None, params), state

    def step_env(self, key, state, action, params):
        move = jnp.where(action == 1, params.step_size, -params.step_size)
        state = EnvState(position=state.position + move, time=state.time + 1)
        terminated = state.position >= params.goal
        reward = terminated.astype(jnp.float32)
        obs = self.observe(key, state, action, params)
        return obs, state, reward, terminated, {}

    def observe(self, key, state, action, params):
        return jnp.stack([state.position, params.goal - state.position])

    @property
    def name(self) -> str:
        return "Corridor-v0"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params):
        return spaces.Discrete(2)

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, (2,), jnp.float32)

    def state_space(self, params):
        return spaces.Dict(
            {
                "position": spaces.Box(-jnp.inf, jnp.inf, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
```

It behaves like any built-in environment:

```python
env = Corridor()
params = env.default_params

key_reset, key_step = jax.random.split(jax.random.key(0))
obs, state = env.reset(key_reset, params)
obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
    key_step, state, jnp.array(1), params
)
```

## The hooks

| Hook | Returns | Notes |
| --- | --- | --- |
| `default_params` | `EnvParams` | Property. |
| `reset_env(key, params)` | `(obs, state)` | Start `state.time` at 0. |
| `step_env(key, state, action, params)` | `(obs, state, reward, terminated, info)` | Increment `state.time`. Report only natural termination. |
| `observe(key, state, action, params)` | `obs` | `action` is `None` after a reset. May return a PyTree. |
| `action_space`, `observation_space`, `state_space` | A space | Take `params`. |
| `name`, `num_actions` | `str`, `int` | Properties. |

A few rules keep an environment compatible with `jax.jit` and `jax.vmap`:

- Do not handle the time limit in `step_env`. The base class derives `truncated` from
  `state.time` and `params.max_steps_in_episode`.
- Use `jnp.where` or `jax.lax.select` in place of Python `if` on array values.
- Keep array shapes independent of `params`. Anything that changes a shape belongs in
  the constructor.
- Return the same `info` structure from every step.

## Registration

Register a factory to construct the environment through `gymnax.make`. The factory
receives the keyword arguments passed to `make`; the class itself is a valid factory.

```python
import gymnax

gymnax.register("Corridor-v0", Corridor)
env, params = gymnax.make("Corridor-v0")
```

Registration is global to the running Python process. Choose a unique ID and register
each factory once.

# Wrappers

Wrappers either modify an environment while keeping the functional gymnax API, or
adapt it to the API of another library. All of them live in `gymnax.wrappers`; see the
[reference](../reference/wrappers/purerl.md) for details.

## Functional wrappers

These wrappers take and return explicit state, so they compose with `jax.jit`,
`jax.vmap` and each other.

| Wrapper | Purpose |
| --- | --- |
| `LogWrapper` | Tracks episode returns and lengths and reports them in `info`. |
| `FlattenObservationWrapper` | Flattens observations to one dimension. |
| `StickyActionWrapper` | Repeats the previous action with a fixed probability. |
| `LegacyStepAPIWrapper` | Restores the five-value `step` of gymnax 0.x. |

```python
import jax
import gymnax
from gymnax.wrappers import FlattenObservationWrapper, LogWrapper

env, params = gymnax.make("Breakout-MinAtar")
env = LogWrapper(FlattenObservationWrapper(env))

key_reset, key_act, key_step = jax.random.split(jax.random.key(0), 3)
obs, state = env.reset(key_reset, params)
action = env.action_space(params).sample(key_act)
obs, state, reward, terminated, truncated, info = env.step(
    key_step, state, action, params
)

print(obs.shape)  # (400,)
print(info["returned_episode"])  # whether this step completed an episode
```

`LogWrapper` wraps the environment state in its own state. Reach the inner state
through `state.env_state`.

Attributes that a wrapper does not define, such as `action_space` or `name`, are
forwarded to the wrapped environment.

## Adapters to other libraries

These adapters store state, parameters and PRNG key internally where the target API
requires it.

| Adapter | Target API | Requires |
| --- | --- | --- |
| `GymnaxToGymWrapper` | `gymnasium.Env` | |
| `GymnaxToVectorGymWrapper` | `gymnasium.vector.VectorEnv` | |
| `GymnaxToDmEnvWrapper` | dm_env-style `TimeStep` | |
| `gymnax.wrappers.brax.GymnaxToBraxWrapper` | `brax.envs.Env` | `brax` |
| `gymnax.wrappers.evojax.GymnaxToEvoJaxTask` | `evojax` `VectorizedTask` | `evojax` |

```python
from gymnax.wrappers import GymnaxToGymWrapper

env, params = gymnax.make("CartPole-v1")
gym_env = GymnaxToGymWrapper(env, params, seed=0)

obs, info = gym_env.reset()
obs, reward, terminated, truncated, info = gym_env.step(gym_env.action_space.sample())
```

The wrapped environment still resets automatically, so calling `reset` after an
episode ends is optional.

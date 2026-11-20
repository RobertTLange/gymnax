"""Compatibility adapters for Gymnax API migrations."""

from functools import partial
from typing import Any

import jax

from gymnax.environments import environment


class LegacyStepAPIWrapper:
    """Expose an environment through the five-value step API of Gymnax 0.x.

    `step` returns `(obs, state, reward, done, info)` with
    `done = terminated | truncated`. The separate flags and the pre-reset
    observation stay available as `info["terminated"]`, `info["truncated"]` and
    `info["final_observation"]`. Intended as a migration aid; new code should use
    the six-value API.

    Args:
        env: Environment to wrap.
    """

    def __init__(self, env: environment.Environment):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[Any, environment.EnvState]:
        """Reset the wrapped environment; identical to its own `reset`."""
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: int | float | jax.Array,
        params: environment.EnvParams | None = None,
    ) -> tuple[Any, environment.EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Step the wrapped environment and merge both flags into `done`."""
        obs, next_state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        done = jax.numpy.logical_or(terminated, truncated)
        return obs, next_state, reward, done, info

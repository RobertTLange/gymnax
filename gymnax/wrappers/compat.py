"""Compatibility adapters for Gymnax API migrations."""

from functools import partial
from typing import Any

import jax

from gymnax.environments import environment


class LegacyStepAPIWrapper:
    """Expose Gymnax 1.0 environments through the removed five-value API."""

    def __init__(self, env: environment.Environment):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[Any, environment.EnvState]:
        """Delegate reset without changing its two-value contract."""
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: int | float | jax.Array,
        params: environment.EnvParams | None = None,
    ) -> tuple[Any, environment.EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Return the historical done flag while retaining terminal metadata."""
        obs, next_state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        done = jax.numpy.logical_or(terminated, truncated)
        return obs, next_state, reward, done, info

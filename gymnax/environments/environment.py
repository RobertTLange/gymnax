"""Base class, state and parameters shared by all Gymnax environments."""

import copy
from functools import partial
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
from flax import struct

TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")
TEnvironment = TypeVar("TEnvironment", bound="Environment")


@struct.dataclass
class EnvState:
    """Base state shared by all environments.

    Environments subclass this dataclass and add their own fields. States are
    immutable PyTrees: `step` returns a new state and never modifies its input.

    Attributes:
        time: Number of steps taken in the current episode.
    """

    time: int


@struct.dataclass
class EnvParams:
    """Base parameters shared by all environments.

    Parameters are passed explicitly to `reset` and `step`, so they can be changed,
    traced by `jax.jit` or batched with `jax.vmap` without rebuilding the
    environment. Use `params.replace(...)` to override individual fields.

    Attributes:
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    max_steps_in_episode: int = 1


class Environment(Generic[TEnvState, TEnvParams]):
    """Abstract base class for environments.

    An environment is a stateless collection of pure functions. Everything that
    changes during an episode lives in an explicit state, and everything that
    configures the dynamics lives in explicit parameters, so `reset` and `step` work
    with `jax.jit`, `jax.vmap` and `jax.lax.scan`.

    Users call `reset` and `step`. Subclasses implement `step_env`, `reset_env`,
    `observe`, `default_params` and the space methods; see the custom environments
    guide for a complete example.
    """

    _supports_transition_gradients = False
    _transition_gradients_enabled = False

    @property
    def supports_transition_gradients(self) -> bool:
        """Whether this environment implements differentiable transitions."""
        return self._supports_transition_gradients

    @property
    def transition_gradients_enabled(self) -> bool:
        """Whether transition observations and states preserve gradients."""
        return self._transition_gradients_enabled

    def with_transition_gradients(self: TEnvironment) -> TEnvironment:
        """Return a copy of the environment that preserves transition gradients.

        By default `step` stops gradients at the returned observation and state. The
        copy keeps them, so `jax.grad` can differentiate through transitions with
        respect to actions, states and parameters. The original environment is left
        unchanged.

        Returns:
            A copy of the environment with transition gradients enabled.

        Raises:
            ValueError: If the environment does not support transition gradients.
        """
        if not self.supports_transition_gradients:
            raise ValueError(f"{self.name} does not support transition gradients")

        configured_environment = copy.copy(self)
        configured_environment._transition_gradients_enabled = True
        return configured_environment

    def _apply_transition_gradient_policy(
        self, observation: Any, state: TEnvState
    ) -> tuple[Any, TEnvState]:
        """Detach transition outputs unless gradients were explicitly enabled."""
        if self.transition_gradients_enabled:
            return observation, state
        return jax.tree.map(jax.lax.stop_gradient, (observation, state))

    @property
    def default_params(self) -> EnvParams:
        """Default parameters of the environment.

        Use `params.replace(...)` to override individual fields before passing the
        parameters to `reset` and `step`.
        """
        return EnvParams()

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> tuple[Any, TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Advance the environment by one step and reset it if the episode ended.

        When the episode terminates or is truncated, the returned observation and
        state already belong to a freshly reset episode. The last observation of the
        finished episode is available as `info["final_observation"]`; use it for
        bootstrapping.

        Args:
            key: PRNG key. It is split internally into one key for the transition
                and one for the potential reset.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters. Defaults to `default_params`.

        Returns:
            A tuple `(obs, state, reward, terminated, truncated, info)`. `terminated`
            marks a natural end of the episode and `truncated` marks the time limit
            `params.max_steps_in_episode`; both can be true at once. `info` holds the
            environment-specific entries plus `terminated`, `truncated` and
            `final_observation`, which equals `obs` unless the episode ended.
        """
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, info = self.step_env(
            key_step, state, action, params
        )
        truncated = self.is_truncated(state_st, params)
        done = jnp.logical_or(terminated, truncated)
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree.map(
            lambda reset_leaf, step_leaf: jax.lax.select(done, reset_leaf, step_leaf),
            obs_re,
            obs_st,
        )
        info = {
            **info,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": obs_st,
        }

        return obs, state, reward, terminated, truncated, info

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: TEnvParams | None = None
    ) -> tuple[Any, TEnvState]:
        """Start a new episode.

        Args:
            key: PRNG key.
            params: Environment parameters. Defaults to `default_params`.

        Returns:
            A tuple `(obs, state)` of the first observation and the initial state.
        """
        if params is None:
            params = self.default_params

        # Reset
        obs, state = self.reset_env(key, params)

        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> tuple[Any, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific transition, implemented by subclasses.

        Unlike `step`, this neither resets the environment nor applies the time
        limit. Time limits must not be folded into `terminated`.

        Args:
            key: PRNG key.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            A tuple `(obs, state, reward, terminated, info)` describing the raw
            transition.
        """
        raise NotImplementedError

    def reset_env(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[jax.Array, TEnvState]:
        """Environment-specific reset, implemented by subclasses.

        Args:
            key: PRNG key.
            params: Environment parameters.

        Returns:
            A tuple `(obs, state)` of the first observation and the initial state.
        """
        raise NotImplementedError

    def observe(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array | None,
        params: TEnvParams,
    ) -> Any:
        """Return the observation for a state and the action that produced it.

        Built-in environments that still implement `get_obs` are adapted here; new
        environments should implement this method directly.

        Args:
            key: PRNG key, for stochastic observations.
            state: Environment state to observe.
            action: Action that produced `state`, or `None` after a reset.
            params: Environment parameters.

        Returns:
            The observation, an array or any PyTree of arrays.
        """
        del action
        return self.get_obs(state, params=params, key=key)

    def get_obs(self, state, params=None, key=None) -> Any:
        """Legacy observation hook retained for existing environment classes."""
        raise NotImplementedError

    def is_truncated(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Check whether the episode reached `params.max_steps_in_episode`."""
        return state.time >= params.max_steps_in_episode

    def is_terminal(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Legacy alias of `is_terminated`; does not include the time limit."""
        return self.is_terminated(state, params)

    def is_terminated(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Check whether the state is a natural end of the episode.

        Optional helper for subclasses; `step` uses the `terminated` flag returned by
        `step_env`.
        """
        raise NotImplementedError

    def discount(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Legacy helper returning 0 for terminated states and 1 otherwise."""
        return jax.lax.select(self.is_terminated(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Name of the environment; for built-ins, the ID passed to `gymnax.make`."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of discrete actions, or the action dimension if continuous."""
        raise NotImplementedError

    def action_space(self, params: TEnvParams):
        """Space of valid actions for the given parameters."""
        raise NotImplementedError

    def observation_space(self, params: TEnvParams):
        """Space of observations for the given parameters."""
        raise NotImplementedError

    def state_space(self, params: TEnvParams):
        """Space of environment states for the given parameters."""
        raise NotImplementedError

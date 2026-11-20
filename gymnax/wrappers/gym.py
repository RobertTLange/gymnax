"""Wrappers exposing Gymnax environments through the Gymnasium API."""

import copy
from typing import Any

import gymnasium as gym
import jax
from gymnasium import core
from gymnasium.vector import utils

from gymnax.environments import environment, spaces


class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
    """Wrap a Gymnax environment as a stateful `gymnasium.Env`.

    The wrapper stores the environment state, parameters and PRNG key, so `step`
    only takes an action. The wrapped environment still resets automatically when
    an episode ends.

    Args:
        env: Gymnax environment to wrap.
        params: Environment parameters. Defaults to `env.default_params`.
        seed: Seed for the PRNG key. Defaults to 0.
    """

    def __init__(
        self,
        env: environment.Environment,
        params: environment.EnvParams | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self._env = copy.deepcopy(env)
        self.env_params = params if params is not None else env.default_params
        self.metadata.update(
            {
                "name": env.name,
                "render_modes": (
                    ["human", "rgb_array"] if hasattr(env, "render") else []
                ),
            }
        )
        self.key: jax.Array = jax.random.key(0)  # Placeholder
        self._seed(seed)
        _, self.env_state = self._env.reset(self.key, self.env_params)

    @property
    def action_space(self):
        """Gymnasium action space for the current parameters."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def observation_space(self):
        """Gymnasium observation space for the current parameters."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    def _seed(self, seed: int | None = None):
        """Set key seed (or use 0)."""
        self.key = jax.random.key(seed or 0)

    def step(
        self, action: core.ActType
    ) -> tuple[core.ObsType, float, bool, bool, dict[Any, Any]]:
        """Take one step and update the stored state.

        Args:
            action: Action to take.

        Returns:
            A tuple `(obs, reward, terminated, truncated, info)`.
        """
        self.key, step_key = jax.random.split(self.key)
        o, self.env_state, r, terminated, truncated, info = self._env.step(
            step_key, self.env_state, action, self.env_params
        )
        return o, r, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: Any | None = None,  # dict
    ) -> tuple[core.ObsType, Any]:  # dict]:
        """Start a new episode.

        Args:
            seed: If given, reseeds the PRNG key before resetting.
            return_info: Accepted for compatibility and ignored.
            options: If it contains the key `"env_params"`, its value replaces the
                stored environment parameters.

        Returns:
            A tuple `(obs, info)` with an empty `info` dictionary.
        """
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.key, reset_key = jax.random.split(self.key)
        o, self.env_state = self._env.reset(reset_key, self.env_params)
        return o, {}

    def render(self, mode="human") -> core.RenderFrame | list[core.RenderFrame] | None:
        """Render the current state if the environment defines `render`.

        Returns `None` otherwise. `mode` is accepted for compatibility and ignored.
        """
        return getattr(self._env, "render", lambda x, y: None)(
            self.env_state, self.env_params
        )


class GymnaxToVectorGymWrapper(gym.vector.VectorEnv):
    """Wrap a Gymnax environment as a stateful `gymnasium.vector.VectorEnv`.

    Runs `num_envs` copies of the environment in parallel with `jax.vmap`. All
    copies share the same parameters and each has its own PRNG key.

    Args:
        env: Gymnax environment to wrap.
        num_envs: Number of environments to run in parallel.
        params: Environment parameters. Defaults to `env.default_params`.
        seed: Seed for the PRNG keys. Defaults to 0.
    """

    def __init__(
        self,
        env: environment.Environment,
        num_envs: int = 1,
        params: environment.EnvParams | None = None,
        seed: int | None = None,
    ):
        self._env = copy.deepcopy(env)
        self.num_envs = num_envs
        self.is_vector_env = True
        self.new_step_api = True
        self.closed = False
        self.viewer = None
        self.key: jax.Array = jax.random.key(0)  # Placeholder
        self._seed(seed)
        # Jit-of-vmap is faster than vmap-of-jit.
        # Map over leading axis of all but env params
        self._env.reset = jax.jit(jax.vmap(self._env.reset, in_axes=(0, None)))
        self._env.step = jax.jit(jax.vmap(self._env.step, in_axes=(0, 0, 0, None)))
        self.env_params = params if params is not None else env.default_params
        _, self.env_state = self._env.reset(self.key, self.env_params)  # Placeholder
        self._batched_key_split = jax.jit(
            jax.vmap(jax.random.split, in_axes=0, out_axes=1)
        )  # Split all keys

    @property
    def single_action_space(self):
        """Gymnasium action space of a single environment."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def single_observation_space(self):
        """Gymnasium observation space of a single environment."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    @property
    def action_space(self):
        """Batched Gymnasium action space of all environments."""
        return utils.batch_space(self.single_action_space, self.num_envs)

    @property
    def observation_space(self):
        """Batched Gymnasium observation space of all environments."""
        return utils.batch_space(self.single_observation_space, self.num_envs)

    def _seed(self, seed: int | None = None):
        """Set key seed (or use 0)."""
        self.key = jax.random.split(
            jax.random.key(seed or 0), self.num_envs
        )  # 1 key per env

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        return_info: bool = False,
        options: Any | None = None,  # dict
    ):  # -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset all environments.

        Args:
            seed: If given, reseeds the PRNG keys before resetting.
            return_info: Accepted for compatibility and ignored.
            options: If it contains the key `"env_params"`, its value replaces the
                stored environment parameters.

        Returns:
            (tuple): A tuple `(obs, info)` with batched observations and an empty
                `info` dictionary.
        """
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.key, reset_key = self._batched_key_split(self.key)  # Split all keys
        o, self.env_state = self._env.reset(reset_key, self.env_params)
        return o, {}

    def step(
        self,
        action,  #: core.ActType
    ):  # -> Tuple[core.ObsType, float, bool, bool, Any]:  # dict]:
        """Step all environments and update the stored states.

        Args:
            action (Array): Batch of actions, one per environment.

        Returns:
            (tuple): A tuple `(obs, reward, terminated, truncated, info)` of batched
                arrays.
        """
        self.key, step_key = self._batched_key_split(self.key)
        o, self.env_state, r, terminated, truncated, info = self._env.step(
            step_key, self.env_state, action, self.env_params
        )
        return o, r, terminated, truncated, info

    def render(self, mode="human") -> core.RenderFrame | list[core.RenderFrame] | None:
        """Render the first environment if the environment defines `render`.

        Returns `None` otherwise. `mode` is accepted for compatibility and ignored.
        """
        return getattr(self._env, "render", lambda x, y: None)(
            jax.tree.map(lambda x: x[0], self.env_state), self.env_params
        )

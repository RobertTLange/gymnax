"""Wrappers for Gymnax environments to be used in Gym."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union


import chex
import gymnasium as gym
from gymnasium import core
from gymnasium.vector import utils
import jax.random
from gymnax.environments import environment
from gymnax.environments import spaces


class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
    """Wrap Gymnax environment as OOP Gym environment."""

    def __init__(
        self,
        env: environment.Environment,
        params: Optional[environment.EnvParams] = None,
        seed: Optional[int] = None,
    ):
        """Wrap Gymnax environment as OOP Gym environment.


        Args:
            env: Gymnax Environment instance
            params: If provided, gymnax EnvParams for environment (otherwise uses
              default)
            seed: If provided, seed for JAX PRNG (otherwise picks 0)
        """
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
        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)
        _, self.env_state = self._env.reset(self.rng, self.env_params)

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)."""
        self.rng = jax.random.PRNGKey(seed or 0)

    def step(
        self, action: core.ActType
    ) -> Tuple[core.ObsType, float, bool, bool, Dict[Any, Any]]:
        """Step environment, follow new step API."""
        self.rng, step_key = jax.random.split(self.rng)
        o, self.env_state, r, d, info = self._env.step(
            step_key, self.env_state, action, self.env_params
        )
        return o, r, d, d, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[Any] = None,  # dict
    ) -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset environment, update parameters and seed if provided."""
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.rng, reset_key = jax.random.split(self.rng)
        o, self.env_state = self._env.reset(reset_key, self.env_params)
        return o, {}

    def render(
        self, mode="human"
    ) -> Optional[Union[core.RenderFrame, List[core.RenderFrame]]]:
        """use underlying environment rendering if it exists, otherwise return None."""
        return getattr(self._env, "render", lambda x, y: None)(
            self.env_state, self.env_params
        )


class GymnaxToVectorGymWrapper(gym.vector.VectorEnv):
    """Wrap Gymnax environment as OOP Gym Vector Environment."""

    def __init__(
        self,
        env: environment.Environment,
        num_envs: int = 1,
        params: Optional[environment.EnvParams] = None,
        seed: Optional[int] = None,
    ):
        """Wrap Gymnax environment as OOP Gym Vector Environment.


        Args:
            env: Gymnax Environment instance
            num_envs: Desired number of environments to run in parallel
            params: If provided, gymnax EnvParams for environment (otherwise uses
              default)
            seed: If provided, seed for JAX PRNG (otherwise picks 0)
        """
        self._env = copy.deepcopy(env)
        self.num_envs = num_envs
        self.is_vector_env = True
        self.new_step_api = True
        self.closed = False
        self.viewer = None
        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)
        # Jit-of-vmap is faster than vmap-of-jit.
        # Map over leading axis of all but env params
        self._env.reset = jax.jit(jax.vmap(self._env.reset, in_axes=(0, None)))
        self._env.step = jax.jit(jax.vmap(self._env.step, in_axes=(0, 0, 0, None)))
        self.env_params = params if params is not None else env.default_params
        _, self.env_state = self._env.reset(self.rng, self.env_params)  # Placeholder
        self._batched_rng_split = jax.jit(
            jax.vmap(jax.random.split, in_axes=0, out_axes=1)
        )  # Split all rng keys

    @property
    def single_action_space(self):
        """Dynamically adjust action space depending on params."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def single_observation_space(self):
        """Dynamically adjust state space depending on params."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params."""
        return utils.batch_space(self.single_action_space, self.num_envs)

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params."""
        return utils.batch_space(self.single_observation_space, self.num_envs)

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)."""
        self.rng = jax.random.split(
            jax.random.PRNGKey(seed or 0), self.num_envs
        )  # 1 RNG per env

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[Any] = None,  # dict
    ):  # -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset environment, update parameters and seed if provided."""
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.rng, reset_key = self._batched_rng_split(self.rng)  # Split all keys
        o, self.env_state = self._env.reset(reset_key, self.env_params)
        return o, {}

    def step(
        self, action  #: core.ActType
    ):  # -> Tuple[core.ObsType, float, bool, bool, Any]:  # dict]:
        """Step environment, follow new step API."""
        self.rng, step_key = self._batched_rng_split(self.rng)
        o, self.env_state, r, d, info = self._env.step(
            step_key, self.env_state, action, self.env_params
        )
        return o, r, d, d, info

    def render(
        self, mode="human"
    ) -> Optional[Union[core.RenderFrame, List[core.RenderFrame]]]:
        """use underlying environment rendering if it exists (for first environment), otherwise return None."""
        return getattr(self._env, "render", lambda x, y: None)(
            jax.tree_map(lambda x: x[0], self.env_state), self.env_params
        )

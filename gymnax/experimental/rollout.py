"""Compiled policy rollouts over batches of keys and parameters."""

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import gymnax


class RolloutWrapper:
    """Roll out a policy in an environment with `jax.lax.scan`.

    `single_rollout` runs one rollout, `batch_rollout` maps it over a batch of PRNG
    keys and `population_rollout` additionally maps it over a batch of policy
    parameters. All three return the same tuple, with one leading axis per mapped
    quantity.

    Args:
        model_forward (Callable | None): Policy function
            `(policy_params, obs, key) -> action`. If `None`, actions are sampled
            uniformly from the action space.
        env_name: Registered Gymnax environment ID. Defaults to `Pendulum-v1`.
        num_env_steps: Number of transitions in each rollout. Defaults to
            `max_steps_in_episode` of the environment parameters.
        env_kwargs: Constructor keyword arguments for `env_name`.
        env_params: Parameter overrides as a mapping of field names to values, or
            a complete parameter object.
        env: Pre-constructed environment, bypassing `gymnax.make`. Cannot be
            combined with `env_name` or `env_kwargs`.
    """

    def __init__(
        self,
        model_forward=None,
        env_name: str | None = None,
        num_env_steps: int | None = None,
        env_kwargs: Any | None = None,
        env_params: Any | None = None,
        env: Any | None = None,
    ):
        self.env_name = env_name
        if env is not None:
            if env_name is not None:
                raise ValueError("env_name cannot be used with an external env")
            if env_kwargs is not None:
                raise ValueError("env_kwargs cannot be used with an external env")
            self.env = env
            self.env_params = _resolve_params(env, env_params)
        else:
            self.env_name = "Pendulum-v1" if env_name is None else env_name
            env_kwargs = {} if env_kwargs is None else env_kwargs
            self.env, default_params = gymnax.make(self.env_name, **env_kwargs)
            self.env_params = _resolve_params(self.env, env_params, default_params)
        self.model_forward = model_forward

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

    @partial(jax.jit, static_argnames=("self",))
    def population_rollout(self, key_eval, policy_params):
        """Roll out a population of policies, each on the same batch of keys.

        Args:
            key_eval (Array): Batch of PRNG keys, one per rollout of each policy.
            policy_params (Any): Policy parameters with a leading population
                axis.

        Returns:
            (tuple): The tuple described in `single_rollout`, with leading axes
                `(population, keys)`.
        """
        # Evaluate population of nets on gymnax task - vmap over key & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(key_eval, policy_params)

    @partial(jax.jit, static_argnames=("self",))
    def batch_rollout(self, key_eval, policy_params):
        """Roll out one policy once per key.

        Args:
            key_eval (Array): Batch of PRNG keys, one per rollout.
            policy_params (Any): Parameters of a single policy.

        Returns:
            (tuple): The tuple described in `single_rollout`, with a leading key
                axis.
        """
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(key_eval, policy_params)

    @partial(jax.jit, static_argnames=("self",))
    def single_rollout(self, key_input, policy_params):
        """Roll out one policy for `num_env_steps` transitions.

        The environment resets automatically, so a rollout can span several
        episodes. Only rewards of the first episode count towards the return.

        Args:
            key_input (Array): PRNG key.
            policy_params (Any): Parameters of a single policy.

        Returns:
            (tuple): A tuple `(obs, action, reward, next_obs, done, cum_return)`.
                The first five entries have a leading time axis of length
                `num_env_steps`; `cum_return` has shape `(1,)` and holds the return
                of the first episode.
        """
        # Reset the environment
        key_reset, key_episode = jax.random.split(key_input)
        obs, state = self.env.reset(key_reset, self.env_params)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jax.random.split(key, 3)
            if self.model_forward is not None:
                action = self.model_forward(policy_params, obs, key_net)
            else:
                action = self.env.action_space(self.env_params).sample(key_net)
            next_obs, next_state, reward, terminated, truncated, _ = self.env.step(
                key_step, state, action, self.env_params
            )
            done = jnp.logical_or(terminated, truncated)
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                key,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                key_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Shape of a single observation."""
        key = jax.random.key(0)
        obs, _ = self.env.reset(key, self.env_params)
        return obs.shape


def _resolve_params(env, env_params, default_params=None):
    """Return complete environment parameters from an object or field overrides."""
    if default_params is None:
        default_params = env.default_params
    if env_params is None:
        return default_params
    if isinstance(env_params, Mapping):
        return default_params.replace(**env_params)
    return env_params

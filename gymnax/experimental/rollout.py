"""Rollout wrapper for gymnax environments."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import gymnax


class RolloutWrapper:
    """Wrapper to define batch evaluation for generation parameters."""

    def __init__(
        self,
        model_forward=None,
        env_name: str = "Pendulum-v1",
        num_env_steps: int | None = None,
        env_kwargs: Any | None = None,
        env_params: Any | None = None,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env_name = env_name
        # Define the RL environment & network forward function
        if env_kwargs is None:
            env_kwargs = {}
        if env_params is None:
            env_params = {}
        self.env, self.env_params = gymnax.make(self.env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.model_forward = model_forward

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

    @partial(jax.jit, static_argnames=("self",))
    def population_rollout(self, key_eval, policy_params):
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over key & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(key_eval, policy_params)

    @partial(jax.jit, static_argnames=("self",))
    def batch_rollout(self, key_eval, policy_params):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(key_eval, policy_params)

    @partial(jax.jit, static_argnames=("self",))
    def single_rollout(self, key_input, policy_params):
        """Rollout a pendulum episode with lax.scan."""
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
            next_obs, next_state, reward, done, _ = self.env.step(
                key_step, state, action, self.env_params
            )
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
            self.env_params.max_steps_in_episode,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        key = jax.random.key(0)
        obs, _ = self.env.reset(key, self.env_params)
        return obs.shape

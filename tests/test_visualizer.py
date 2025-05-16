"""Tests for the visualizer."""

import os

import jax
import jax.numpy as jnp

import gymnax
from gymnax.visualize import visualizer


def test_visualizer(viz_env_name: str):
    """Tests the visualizer."""
    key = jax.random.key(0)
    env, env_params = gymnax.make(viz_env_name)
    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    _, env_state = env.reset(key_reset, env_params)
    while True:
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)
        action = env.action_space(env_params).sample(key_act)
        next_obs, next_env_state, reward, done, _ = env.step(
            key_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            _ = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = visualizer.Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate("anim.gif")
    assert os.path.exists("anim.gif")

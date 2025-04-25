"""Tests for the visualizer."""

import os

import jax
import jax.numpy as jnp

import gymnax
from gymnax.visualize import visualizer


def test_visualizer(viz_env_name: str):
    """Tests the visualizer."""
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make(viz_env_name)
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    _, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, _ = env.step(
            rng_step, env_state, action, env_params
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

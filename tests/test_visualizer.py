"""Tests for the visualizer."""

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

import gymnax
from gymnax.visualize import visualizer


def rollout_states(env_name: str, num_steps: int = 2):
    """Create a short state sequence without requiring an episode to terminate."""
    key = jax.random.key(0)
    env, env_params = gymnax.make(env_name)
    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    _, env_state = env.reset(key_reset, env_params)
    for _ in range(num_steps):
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)
        action = env.action_space(env_params).sample(key_act)
        _, env_state, reward, _, _ = env.step(key_step, env_state, action, env_params)
        reward_seq.append(reward)
    return env, env_params, state_seq, jnp.cumsum(jnp.array(reward_seq))


@pytest.mark.parametrize(
    "env_name", ["Catch-bsuite", "MetaMaze-misc", "SpaceInvaders-MinAtar"]
)
def test_native_visualizer_writes_temporary_animation(env_name: str, tmp_path):
    """Supported native visualizer paths render without leaving repository files."""
    env, env_params, state_seq, rewards = rollout_states(env_name)
    animation_path = tmp_path / f"{env_name}.gif"
    vis = visualizer.Visualizer(env, env_params, state_seq, rewards)
    try:
        vis.animate(str(animation_path))
        assert animation_path.is_file()
        assert animation_path.stat().st_size > 0
    finally:
        plt.close(vis.fig)


@pytest.mark.parametrize("env_name", ["CartPole-v1", "Acrobot-v1", "Pendulum-v1"])
@pytest.mark.xfail(reason="WP2/#57: Gymnasium visualizer compatibility is pending.")
def test_gymnasium_visualizer_reproduction_probe(env_name: str, tmp_path):
    """Record the modern Gymnasium rendering regression for WP2."""
    env, env_params, state_seq, rewards = rollout_states(env_name)
    animation_path = tmp_path / f"{env_name}.gif"
    vis = visualizer.Visualizer(env, env_params, state_seq, rewards)
    try:
        vis.animate(str(animation_path))
        assert animation_path.is_file()
    finally:
        plt.close(vis.fig)

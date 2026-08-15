"""Tests for the visualizer."""

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import gymnax
from gymnax.visualize import vis_gym, visualizer

CLASSIC_CONTROL_ENVS = [
    "Acrobot-v1",
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
]


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
        _, env_state, reward, _, _, _ = env.step(
            key_step, env_state, action, env_params
        )
        reward_seq.append(reward)
    return env, env_params, state_seq, jnp.cumsum(jnp.array(reward_seq))


@pytest.mark.parametrize(
    "env_name",
    ["Catch-bsuite", "MetaMaze-misc", "Seaquest-MinAtar", "SpaceInvaders-MinAtar"],
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
        assert not plt.fignum_exists(vis.fig.number)
    finally:
        plt.close(vis.fig)


@pytest.mark.parametrize("env_name", CLASSIC_CONTROL_ENVS)
def test_gymnasium_visualizer_writes_temporary_animation(env_name: str, tmp_path):
    """Modern Gymnasium classic-control paths render without ffmpeg."""
    env, env_params, state_seq, rewards = rollout_states(env_name)
    animation_path = tmp_path / f"{env_name}.gif"
    vis = visualizer.Visualizer(env, env_params, state_seq, rewards)
    try:
        vis.animate(str(animation_path))
        assert animation_path.is_file()
        assert animation_path.stat().st_size > 0
        assert not plt.fignum_exists(vis.fig.number)
    finally:
        plt.close(vis.fig)


@pytest.mark.parametrize("env_name", CLASSIC_CONTROL_ENVS)
def test_gymnasium_frame_adapter_returns_rgb_array(env_name: str):
    """Classic-control frames use the current Gymnasium rendering contract."""
    env, env_params, state_seq, _ = rollout_states(env_name)

    frame = vis_gym.render_gym_frame(env.name, state_seq[0], env_params)

    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[-1] == 3
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0


@pytest.mark.parametrize("env_name", CLASSIC_CONTROL_ENVS)
def test_gymnasium_frame_adapter_transfers_state_params_and_closes(
    env_name, monkeypatch
):
    """Adapters transfer all render inputs below Gymnasium wrappers."""
    env, env_params, state_seq, _ = rollout_states(env_name)

    class FakeGymEnv:
        def __init__(self):
            self.unwrapped = type("Unwrapped", (), {})()
            self.closed = False

        def reset(self):
            return None, {}

        def render(self):
            return np.zeros((2, 3, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    fake_env = FakeGymEnv()
    make_calls = []
    monkeypatch.setattr(
        vis_gym.gym,
        "make",
        lambda *args, **kwargs: make_calls.append((args, kwargs)) or fake_env,
    )

    frame = vis_gym.render_gym_frame(env_name, state_seq[0], env_params)

    assert frame.shape == (2, 3, 3)
    assert make_calls == [((env_name,), {"render_mode": "rgb_array"})]
    np.testing.assert_allclose(
        fake_env.unwrapped.state, vis_gym.get_gym_state(state_seq[0], env_name)
    )
    for gym_attr, gymnax_attr in vis_gym._PARAMETER_MAPPINGS[env_name].items():
        assert getattr(fake_env.unwrapped, gym_attr) == pytest.approx(
            float(getattr(env_params, gymnax_attr))
        )
    if env_name == "Pendulum-v1":
        np.testing.assert_allclose(
            fake_env.unwrapped.last_u,
            np.asarray(state_seq[0].last_u, dtype=np.float64),
        )
    assert fake_env.closed


def test_gymnasium_frame_adapter_rejects_unsupported_environment():
    """The adapter does not silently choose an unrelated Gymnasium renderer."""
    with pytest.raises(ValueError, match="Unsupported Gymnasium visualizer"):
        vis_gym.render_gym_frame("Catch-bsuite", object(), object())

"""Baseline reproduction-matrix smoke tests for supported integrations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gymnax
from gymnax.environments.bsuite import mnist


@pytest.fixture
def local_mnist_dataset(monkeypatch):
    """Keep the registered MNIST smoke test deterministic and offline."""

    images = np.zeros((2, 28, 28), dtype=np.uint8)
    labels = np.array([0, 1], dtype=np.uint8)

    def load_mnist():
        return (images, labels), (images, labels)

    monkeypatch.setattr(mnist.load_mnist, "load_mnist", load_mnist)


@pytest.mark.skipif(
    jax.config.x64_enabled,
    reason="the complete registered-environment smoke suite is default precision",
)
@pytest.mark.parametrize("env_name", gymnax.registered_envs)
def test_registered_environment_resets_and_steps_when_jitted(
    env_name, local_mnist_dataset
):
    """Every registered environment supports make, reset, action sampling, and step."""
    env, params = gymnax.make(env_name)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    key, reset_key, action_key, step_key = jax.random.split(jax.random.key(0), 4)

    observation, state = reset(reset_key, params)
    action = env.action_space(params).sample(action_key)
    next_observation, next_state, reward, terminated, truncated, info = step(
        step_key, state, action, params
    )
    done = jnp.logical_or(terminated, truncated)

    assert observation.shape == next_observation.shape
    assert reward.shape == ()
    assert done.shape == ()
    assert info
    assert {"terminated", "truncated", "final_observation"} <= info.keys()
    assert done == jnp.logical_or(terminated, truncated)
    assert next_observation.shape == info["final_observation"].shape
    jax.block_until_ready((next_observation, next_state, reward, done))


def test_brax_adapter_resets_and_steps_when_available():
    """The optional Brax adapter supports its direct reset and step contract."""
    pytest.importorskip("brax")
    from gymnax.wrappers.brax import GymnaxToBraxWrapper

    env, params = gymnax.make("CartPole-v1")
    adapter = GymnaxToBraxWrapper(env)
    reset = jax.jit(adapter.reset)
    step = jax.jit(adapter.step)
    key, reset_key, action_key = jax.random.split(jax.random.key(0), 3)

    state = reset(reset_key, params)
    action = env.action_space(params).sample(action_key)
    next_state = step(state, action, params)

    assert state.obs.shape == next_state.obs.shape
    assert next_state.reward.shape == ()
    assert next_state.done.shape == ()
    jax.block_until_ready(next_state)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="requires JAX x64 mode")
def test_cartpole_jitted_step_under_x64():
    """A classic-control baseline remains executable with JAX x64 enabled."""
    env, params = gymnax.make("CartPole-v1")
    key, reset_key, action_key, step_key = jax.random.split(jax.random.key(0), 4)
    observation, state = jax.jit(env.reset)(reset_key, params)
    action = env.action_space(params).sample(action_key)
    next_observation, *_ = jax.jit(env.step)(step_key, state, action, params)

    assert observation.dtype == jnp.float64
    assert next_observation.dtype == jnp.float64


@pytest.mark.skipif(not jax.config.x64_enabled, reason="requires JAX x64 mode")
def test_acrobot_jitted_step_under_x64():
    """Acrobot auto-reset preserves a consistent int32 timestep under x64."""
    env, params = gymnax.make("Acrobot-v1")
    params = params.replace(max_steps_in_episode=1)
    key, reset_key, action_key, step_key = jax.random.split(jax.random.key(0), 4)
    _, state = jax.jit(env.reset)(reset_key, params)
    action = env.action_space(params).sample(action_key)
    _, next_state, _, terminated, truncated, _ = jax.jit(env.step)(
        step_key, state, action, params
    )
    done = jnp.logical_or(terminated, truncated)

    assert state.time.dtype == jnp.int32
    assert done
    assert next_state.time.dtype == jnp.int32
    assert next_state.time == 0

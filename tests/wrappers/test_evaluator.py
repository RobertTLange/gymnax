"""Tests for evaluator wrapper."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

import gymnax
from gymnax.experimental import rollout


class MLP(nn.Module):
    """Simple MLP Wrapper with flexible output head."""

    @nn.compact
    def __call__(self, x, key):
        # Loop over dense layers in forward pass
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        x = nn.tanh(x)
        return x


def test_rollout():
    """Test rollout wrapper."""
    key = jax.random.key(0)
    model = MLP()
    pholder = jnp.zeros((3,))
    num_env_steps = 150
    policy_params = model.init(
        key,
        x=pholder,
        key=key,
    )
    manager = rollout.RolloutWrapper(
        model.apply, env_name="Pendulum-v1", num_env_steps=num_env_steps
    )

    # Test simple single episode rollout
    obs, _, _, _, _, _ = manager.single_rollout(key, policy_params)
    assert obs.shape == (num_env_steps, 3)

    # Test multiple rollouts for same network (different random numbers)
    key_batch = jax.random.split(key, 10)
    obs, _, _, _, _, _ = manager.batch_rollout(key_batch, policy_params)
    assert obs.shape == (10, num_env_steps, 3)

    # Test multiple rollouts for different networks
    batch_params = jax.tree.map(
        lambda x: jnp.tile(x, (5, 1)).reshape(5, *x.shape), policy_params
    )
    # print(jax.tree.map(lambda x: x.shape, policy_params))
    # print(jax.tree.map(lambda x: x.shape, batch_params))
    (
        obs,
        _,
        _,
        _,
        _,
        _,
    ) = manager.population_rollout(key_batch, batch_params)
    assert obs.shape == (5, 10, num_env_steps, 3)


def test_rollout_accepts_an_environment_and_complete_parameters():
    """External environments bypass registration and retain their parameters."""
    env = gymnax.environments.CartPole()
    params = env.default_params
    params = params.replace(max_steps_in_episode=4)
    manager = rollout.RolloutWrapper(
        env=env,
        env_params=params,
        num_env_steps=3,
    )

    observations, *_ = manager.single_rollout(jax.random.key(0), None)

    assert manager.env is env
    assert manager.env_params == params
    assert observations.shape == (3, 4)


def test_rollout_rejects_constructor_options_for_an_external_environment():
    """An injected environment cannot also receive registry constructor options."""
    env, _ = gymnax.make("CartPole-v1")

    with pytest.raises(ValueError, match="env_kwargs"):
        rollout.RolloutWrapper(env=env, env_kwargs={"unused": True})
    with pytest.raises(ValueError, match="env_name"):
        rollout.RolloutWrapper(env=env, env_name="CartPole-v1")


def test_rollout_preserves_named_environment_parameter_overrides():
    """The original named-environment API still accepts parameter mappings."""
    manager = rollout.RolloutWrapper(
        env_name="CartPole-v1",
        env_params={"max_steps_in_episode": 3},
    )

    assert manager.env_params.max_steps_in_episode == 3


def test_deep_sea_rollout_defaults_to_grid_size():
    """Naturally terminal DeepSea rollouts avoid masked extra transitions."""
    env = gymnax.environments.DeepSea(size=3)
    manager = rollout.RolloutWrapper(env=env)

    observations, _, _, _, done, _ = manager.single_rollout(jax.random.key(0), None)

    assert manager.num_env_steps == 3
    assert observations.shape == (3, 3, 3)
    assert jnp.array_equal(done, jnp.array([False, False, True]))

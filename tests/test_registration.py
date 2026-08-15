"""Tests for public environment registration."""

import jax
import jax.numpy as jnp
import pytest
from flax import struct

import gymnax
from gymnax.environments import environment, spaces


@struct.dataclass
class CustomParams(environment.EnvParams):
    reward: float = 1.0


class CustomEnvironment(environment.Environment[environment.EnvState, CustomParams]):
    """Small environment used to exercise runtime registration."""

    def __init__(self, reward: float = 1.0):
        self._reward = reward

    @property
    def default_params(self) -> CustomParams:
        return CustomParams(reward=self._reward)

    def reset_env(self, key, params):
        del key, params
        state = environment.EnvState(time=0)
        return jnp.array(0, dtype=jnp.int32), state

    def step_env(self, key, state, action, params):
        del key, action
        state = state.replace(time=state.time + 1)
        return jnp.array(0, dtype=jnp.int32), state, params.reward, jnp.array(False), {}

    def is_terminated(self, state, params):
        del state, params
        return jnp.array(False)

    def action_space(self, params):
        del params
        return spaces.Discrete(1)

    def observation_space(self, params):
        del params
        return spaces.Discrete(1)

    def state_space(self, params):
        return spaces.Dict({"time": spaces.Discrete(params.max_steps_in_episode + 1)})


def _isolated_registry(monkeypatch):
    """Prevent runtime registration from leaking between tests."""
    registration = gymnax.registration
    monkeypatch.setattr(registration, "_registry", dict(registration._registry))
    monkeypatch.setattr(registration, "registered_envs", list(gymnax.registered_envs))
    monkeypatch.setattr(gymnax, "registered_envs", registration.registered_envs)


def test_registered_environment_order_remains_compatible():
    """Built-in IDs retain their public order and append FrozenLake."""
    assert gymnax.registered_envs[:3] == ["CartPole-v1", "Pendulum-v1", "Acrobot-v1"]
    assert gymnax.registered_envs[-1] == "FrozenLake-misc"


def test_register_factory_makes_fresh_custom_environments(monkeypatch):
    """Registered factories receive make kwargs and are invoked per make call."""
    env_id = "TestCustom-v0"
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return CustomEnvironment(**kwargs)

    _isolated_registry(monkeypatch)

    gymnax.register(env_id, factory)
    first, first_params = gymnax.make(env_id, reward=2.0)
    second, second_params = gymnax.make(env_id, reward=3.0)

    assert gymnax.registered_envs is gymnax.registration.registered_envs
    assert env_id in gymnax.registered_envs
    assert first is not second
    assert calls == [{"reward": 2.0}, {"reward": 3.0}]
    assert first_params.reward == 2.0
    assert second_params.reward == 3.0

    _, state = jax.jit(first.reset)(jax.random.key(0), first_params)
    _, _, reward, _, _, _ = jax.jit(first.step)(
        jax.random.key(1), state, jnp.array(0), first_params
    )
    assert reward == 2.0


def test_register_rejects_duplicate_ids_and_invalid_factories(monkeypatch):
    """Registration fails before it can replace an existing environment."""
    _isolated_registry(monkeypatch)

    with pytest.raises(ValueError, match="already registered"):
        gymnax.register("CartPole-v1", CustomEnvironment)
    with pytest.raises(TypeError, match="callable"):
        gymnax.register("NotCallable-v0", object())
    with pytest.raises(TypeError, match="must be a string"):
        gymnax.register(1, CustomEnvironment)
    with pytest.raises(ValueError, match="must not be empty"):
        gymnax.register("", CustomEnvironment)
    with pytest.raises(ValueError, match="not in registered"):
        gymnax.make("NotRegistered-v0")


def test_make_rejects_factories_that_do_not_create_environments(monkeypatch):
    """Factory errors identify invalid results instead of leaking attribute errors."""
    _isolated_registry(monkeypatch)
    gymnax.register("InvalidFactory-v0", object)

    with pytest.raises(TypeError, match="must return a Gymnax Environment"):
        gymnax.make("InvalidFactory-v0")

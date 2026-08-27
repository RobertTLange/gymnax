"""Regression tests for canonical bsuite DeepSea behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gymnax.environments.bsuite.deep_sea import DeepSea, EnvParams, EnvState


@pytest.mark.parametrize("size", [1, 3, 8])
def test_default_episode_horizon_matches_size(size):
    """Default rollouts stop at DeepSea's constructor-defined horizon."""
    env = DeepSea(size=size)

    assert env.default_params.max_steps_in_episode == size


def test_natural_termination_coincides_with_default_horizon():
    """The final DeepSea transition reaches both terminal conditions."""
    env = DeepSea(size=3)
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)

    for step in range(2):
        _, state, _, terminated, truncated, _ = env.step(
            jax.random.key(step + 1), state, 0, params
        )
        assert not terminated
        assert not truncated

    _, _, _, terminated, truncated, _ = env.step(jax.random.key(3), state, 0, params)

    assert terminated
    assert truncated


def _key_where_right_move_fails(size: int) -> jax.Array:
    for seed in range(100):
        key = jax.random.key(seed)
        transition_key = jax.random.split(key)[1]
        if not bool(jax.random.uniform(transition_key) > 1 / size):
            return key
    raise AssertionError("No failing windy transition key found")


def test_default_mapping_is_sampled_once_from_constructor_key():
    """Canonical DeepSea samples one randomized map per environment instance."""
    mapping_key = jax.random.key(7)
    env = DeepSea(size=4, action_mapping_key=mapping_key)
    expected_mapping = jax.random.bernoulli(mapping_key, 0.5, (4, 4))

    _, first_state = env.reset(jax.random.key(1), env.default_params)
    _, second_state = env.reset(jax.random.key(2), env.default_params)

    assert env.default_params.randomize_actions
    assert jnp.array_equal(first_state.action_mapping, expected_mapping)
    assert jnp.array_equal(second_state.action_mapping, expected_mapping)


def test_debug_mapping_uses_action_one_for_right_everywhere():
    """Turning off action randomization retains bsuite's debug map."""
    env = DeepSea(size=3, action_mapping_key=jax.random.key(3))
    params = env.default_params.replace(randomize_actions=False)

    _, state = env.reset(jax.random.key(4), params)

    assert jnp.array_equal(state.action_mapping, jnp.ones((3, 3), dtype=jnp.int32))


@pytest.mark.parametrize("deterministic", [True, False])
def test_legacy_sample_action_map_resamples_independent_of_dynamics(deterministic):
    """Legacy resampling remains available without changing transition mode."""
    env = DeepSea(size=4, action_mapping_key=jax.random.key(0))
    params = env.default_params.replace(
        deterministic=deterministic,
        randomize_actions=False,
        sample_action_map=True,
    )
    first_key = jax.random.key(11)
    second_key = jax.random.key(12)

    _, first_state = env.reset(first_key, params)
    _, second_state = env.reset(second_key, params)

    assert jnp.array_equal(
        first_state.action_mapping,
        jax.random.bernoulli(first_key, 0.5, (4, 4)),
    )
    assert jnp.array_equal(
        second_state.action_mapping,
        jax.random.bernoulli(second_key, 0.5, (4, 4)),
    )


def test_windy_right_action_pays_cost_when_transition_fails():
    """bsuite charges a right-action cost even when wind blocks movement."""
    env = DeepSea(size=2)
    params = EnvParams(
        deterministic=False,
        randomize_actions=False,
        unscaled_move_cost=0.01,
    )
    state = EnvState(
        row=0,
        column=0,
        bad_episode=False,
        total_bad_episodes=0,
        denoised_return=0,
        optimal_return=0.24,
        action_mapping=jnp.ones((2, 2), dtype=jnp.int32),
        time=0,
    )

    failing_key = _key_where_right_move_fails(2)
    _, next_state, reward, _, _ = env.step_env(failing_key, state, 1, params)

    assert next_state.column == 0
    assert reward == pytest.approx(-0.005)


def test_windy_endpoint_reward_includes_gaussian_noise():
    """Only windy DeepSea adds endpoint Gaussian reward noise."""
    env = DeepSea(size=2)
    params = EnvParams(deterministic=False, randomize_actions=False)
    state = EnvState(
        row=1,
        column=0,
        bad_episode=False,
        total_bad_episodes=0,
        denoised_return=0,
        optimal_return=0.24,
        action_mapping=jnp.ones((2, 2), dtype=jnp.int32),
        time=1,
    )
    key = jax.random.key(31)
    expected_noise = jax.random.normal(jax.random.split(key)[0])

    _, _, reward, _, _ = env.step_env(key, state, 0, params)

    assert reward == pytest.approx(expected_noise)


@pytest.mark.parametrize("actions", [(1, 1, 1), (1, 0, 1), (0, 1, 0)])
def test_deterministic_episodes_match_bsuite_reference(actions):
    """Debug-mode deterministic trajectories retain bsuite's exact semantics."""
    bsuite_deep_sea = pytest.importorskip("bsuite.environments.deep_sea")
    with pytest.warns(UserWarning, match="debug mode"):
        reference_env = bsuite_deep_sea.DeepSea(
            size=3,
            deterministic=True,
            unscaled_move_cost=0.01,
            randomize_actions=False,
        )
    gymnax_env = DeepSea(size=3, action_mapping_key=jax.random.key(0))
    params = gymnax_env.default_params.replace(randomize_actions=False)

    reference_timestep = reference_env.reset()
    _, state = gymnax_env.reset(jax.random.key(1), params)
    for step, action in enumerate(actions):
        reference_timestep = reference_env.step(action)
        observation, state, reward, done, _ = gymnax_env.step_env(
            jax.random.key(step + 2), state, action, params
        )

        assert jnp.array_equal(observation, reference_timestep.observation)
        assert float(reward) == pytest.approx(reference_timestep.reward)
        assert bool(done) is reference_timestep.last()


def test_randomized_mapping_trajectory_matches_bsuite_reference():
    """The fixed randomized map drives the same deterministic bsuite trajectory."""
    bsuite_deep_sea = pytest.importorskip("bsuite.environments.deep_sea")
    gymnax_env = DeepSea(size=3, action_mapping_key=jax.random.key(17))
    params = gymnax_env.default_params
    _, state = gymnax_env.reset(jax.random.key(1), params)
    reference_env = bsuite_deep_sea.DeepSea(
        size=3,
        deterministic=True,
        unscaled_move_cost=0.01,
        randomize_actions=True,
        mapping_seed=0,
    )
    reference_env._action_mapping = np.asarray(state.action_mapping)

    reference_env.reset()
    for step, action in enumerate((0, 1, 1)):
        reference_timestep = reference_env.step(action)
        observation, state, reward, done, _ = gymnax_env.step_env(
            jax.random.key(step + 2), state, action, params
        )

        assert jnp.array_equal(observation, reference_timestep.observation)
        assert float(reward) == pytest.approx(reference_timestep.reward)
        assert bool(done) is reference_timestep.last()


def test_autoreset_preserves_cumulative_bsuite_metrics():
    """Episode resets do not discard DeepSea's cumulative reference counters."""
    env = DeepSea(size=1)
    params = env.default_params.replace(randomize_actions=False)
    _, state = env.reset(jax.random.key(0), params)

    _, state, _, first_terminated, first_truncated, _ = env.step(
        jax.random.key(1), state, 1, params
    )
    _, state, _, second_terminated, second_truncated, _ = env.step(
        jax.random.key(2), state, 0, params
    )
    first_done = jnp.logical_or(first_terminated, first_truncated)
    second_done = jnp.logical_or(second_terminated, second_truncated)

    assert first_done and second_done
    assert state.denoised_return == 1
    assert state.total_bad_episodes == 1

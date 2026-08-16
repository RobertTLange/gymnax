"""Tests for opt-in differentiable environment transitions."""

import chex
import jax
import jax.numpy as jnp
import pytest

import gymnax
from gymnax.environments.classic_control import pendulum
from gymnax.environments.misc import point_robot
from gymnax.wrappers import FlattenObservationWrapper

SUPPORTED_ENVIRONMENTS = (
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "PointRobot-misc",
    "Reacher-misc",
    "Swimmer-misc",
)


def _pendulum_fixture():
    env, params = gymnax.make("Pendulum-v1")
    state = pendulum.EnvState(
        theta=jnp.array(0.5),
        theta_dot=jnp.array(0.2),
        last_u=jnp.array(0.0),
        time=0,
    )
    action = jnp.array([0.4])
    return env, params, state, action, jax.random.key(0)


def _pendulum_derivatives(env, params, state, action, key):
    observation_jacobian = jax.jacrev(
        lambda current_action: env.step(key, state, current_action, params)[0]
    )(action)
    theta_dot_jacobian = jax.jacrev(
        lambda current_action: env.step(key, state, current_action, params)[1].theta_dot
    )(action)
    reward_gradient = jax.grad(
        lambda current_action: env.step(key, state, current_action, params)[2]
    )(action)
    return observation_jacobian, theta_dot_jacobian, reward_gradient


def test_configured_copy_enables_supported_environment_without_mutating_original():
    env, _ = gymnax.make("Pendulum-v1")

    differentiable_env = env.with_transition_gradients()

    assert differentiable_env is not env
    assert env.supports_transition_gradients
    assert not env.transition_gradients_enabled
    assert differentiable_env.transition_gradients_enabled


def test_discrete_environment_rejects_transition_gradients():
    env, _ = gymnax.make("CartPole-v1")

    with pytest.raises(ValueError, match="CartPole-v1.*transition gradients"):
        env.with_transition_gradients()


def test_transition_gradient_capability_is_read_only():
    env, _ = gymnax.make("CartPole-v1")

    with pytest.raises(AttributeError):
        env.supports_transition_gradients = True


def test_default_pendulum_preserves_stopped_transition_gradients():
    env, params, state, action, key = _pendulum_fixture()

    observation_jacobian, theta_dot_jacobian, reward_gradient = _pendulum_derivatives(
        env, params, state, action, key
    )

    chex.assert_trees_all_close(
        (observation_jacobian, theta_dot_jacobian, reward_gradient),
        (jnp.zeros((3, 1)), jnp.zeros((1,)), jnp.array([-0.0008])),
        atol=1e-7,
    )


def test_opt_in_pendulum_matches_analytic_transition_gradients():
    env, params, state, action, key = _pendulum_fixture()
    env = env.with_transition_gradients()

    observation_jacobian, theta_dot_jacobian, reward_gradient = _pendulum_derivatives(
        env, params, state, action, key
    )
    next_theta = env.step(key, state, action, params)[1].theta
    expected_observation_jacobian = jnp.array(
        [
            [-jnp.sin(next_theta) * 0.0075],
            [jnp.cos(next_theta) * 0.0075],
            [0.15],
        ]
    )

    chex.assert_trees_all_close(
        (observation_jacobian, theta_dot_jacobian, reward_gradient),
        (expected_observation_jacobian, jnp.array([0.15]), jnp.array([-0.0008])),
        atol=1e-7,
    )


def test_opt_in_pendulum_propagates_gradients_across_steps():
    env, params, state, action, key = _pendulum_fixture()
    env = env.with_transition_gradients()
    second_action = jnp.array([0.0])

    def second_reward(first_action):
        _, next_state, _, _, _, _ = env.step(key, state, first_action, params)
        return env.step(key, next_state, second_action, params)[2]

    first_action_gradient = jax.grad(second_reward)(action)

    assert jnp.any(jnp.abs(first_action_gradient) > 1e-7)


def test_gradient_mode_does_not_change_transition_values():
    env, params, state, action, key = _pendulum_fixture()

    default_transition = env.step(key, state, action, params)
    differentiable_transition = env.with_transition_gradients().step(
        key, state, action, params
    )

    chex.assert_trees_all_close(default_transition, differentiable_transition)


def test_transition_gradients_compose_with_jit_and_vmap():
    env, params, state, _, key = _pendulum_fixture()
    env = env.with_transition_gradients()
    actions = jnp.array([[0.2], [0.4]])

    observation_jacobian = jax.jit(
        jax.vmap(
            jax.jacrev(
                lambda current_action: env.step(key, state, current_action, params)[0]
            )
        )
    )(actions)

    assert observation_jacobian.shape == (2, 3, 1)
    assert jnp.all(jnp.isfinite(observation_jacobian))
    assert jnp.all(jnp.any(jnp.abs(observation_jacobian) > 1e-7, axis=(1, 2)))


def test_autoreset_keeps_gradient_on_final_observation():
    env, params, state, action, key = _pendulum_fixture()
    env = env.with_transition_gradients()
    params = params.replace(max_steps_in_episode=1)

    reset_observation_jacobian = jax.jacrev(
        lambda current_action: env.step(key, state, current_action, params)[0]
    )(action)
    reset_state_jacobian = jax.jacrev(
        lambda current_action: env.step(key, state, current_action, params)[1].theta_dot
    )(action)
    final_observation_jacobian = jax.jacrev(
        lambda current_action: env.step(key, state, current_action, params)[5][
            "final_observation"
        ]
    )(action)

    assert jnp.all(reset_observation_jacobian == 0)
    assert jnp.all(reset_state_jacobian == 0)
    assert jnp.any(jnp.abs(final_observation_jacobian) > 1e-7)


@pytest.mark.parametrize("env_name", SUPPORTED_ENVIRONMENTS)
def test_supported_environments_expose_finite_observation_gradients(env_name):
    env, params = gymnax.make(env_name)
    env = env.with_transition_gradients()
    reset_key, step_key = jax.random.split(jax.random.key(1))
    _, state = env.reset(reset_key, params)
    action = jnp.zeros(env.action_space(params).shape)

    observation_jacobian = jax.jacrev(
        lambda current_action: env.step_env(step_key, state, current_action, params)[0]
    )(action)

    assert env.supports_transition_gradients
    assert jnp.all(jnp.isfinite(observation_jacobian))
    assert jnp.any(jnp.abs(observation_jacobian) > 1e-7)


def test_point_robot_preserves_position_dynamics_gradients_away_from_respawn():
    env, params = gymnax.make("PointRobot-misc")
    env = env.with_transition_gradients()
    state = point_robot.EnvState(
        last_action=jnp.zeros(2),
        last_reward=jnp.array(0.0),
        pos=jnp.zeros(2),
        goal=jnp.array([1.0, 0.0]),
        goals_reached=0,
        time=0.0,
    )
    action = jnp.array([0.05, -0.05])
    key = jax.random.key(2)

    observation_jacobian = jax.jacrev(
        lambda current_action: env.step_env(key, state, current_action, params)[0][:2]
    )(action)
    position_jacobian = jax.jacrev(
        lambda current_action: env.step_env(key, state, current_action, params)[1].pos
    )(action)

    assert jnp.linalg.norm(state.goal - state.pos) > params.goal_radius
    chex.assert_trees_all_close(
        (observation_jacobian, position_jacobian),
        (jnp.eye(2), jnp.eye(2)),
    )


def test_configured_environment_composes_with_observation_wrapper():
    env, params, state, action, key = _pendulum_fixture()
    wrapped_env = FlattenObservationWrapper(env.with_transition_gradients())

    observation_jacobian = jax.jacrev(
        lambda current_action: wrapped_env.step(key, state, current_action, params)[0]
    )(action)

    assert jnp.any(jnp.abs(observation_jacobian) > 1e-7)

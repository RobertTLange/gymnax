"""Tests for the JAX Seaquest MinAtar environment."""

import jax
import jax.numpy as jnp
import numpy as np
from minatar import environment as minatar_environment

import gymnax


def test_seaquest_is_registered_and_resets():
    """Seaquest exposes the standard Gymnax environment contract."""
    environment, params = gymnax.make("Seaquest-MinAtar")

    observation, state = environment.reset(jax.random.key(0), params)

    assert environment.name == "Seaquest-MinAtar"
    assert environment.num_actions == 6
    assert observation.shape == (10, 10, 10)
    assert observation.dtype == jnp.float32
    assert int(state.oxygen) == params.max_oxygen
    assert int(state.sub_x) == 5
    assert int(state.sub_y) == 0
    assert bool(environment.state_space(params).contains(state))


def test_seaquest_firing_creates_and_moves_a_bullet():
    """A fired bullet leaves the right-facing submarine on the next frame."""
    environment, params = gymnax.make("Seaquest-MinAtar")
    _, state = environment.reset(jax.random.key(0), params)
    state = state.replace(sub_or=jnp.array(True))

    _, next_state, reward, done, _ = environment.step_env(
        jax.random.key(1), state, jnp.array(5), params
    )

    assert int(next_state.f_bullet_count) == 1
    assert jnp.array_equal(next_state.f_bullets[0], jnp.array([6, 0, 1]))
    assert float(reward) == 0.0
    assert not bool(done)


def test_seaquest_step_is_jittable():
    """Seaquest supports compiled functional rollouts."""
    environment, params = gymnax.make("Seaquest-MinAtar")
    _, state = environment.reset(jax.random.key(0), params)

    step = jax.jit(environment.step_env, static_argnums=())
    observation, next_state, _, _, _ = step(
        jax.random.key(2), state, jnp.array(0), params
    )

    assert observation.shape == (10, 10, 10)
    assert int(next_state.time) == 1


def test_seaquest_keeps_extra_divers_when_rescue_capacity_is_full():
    """Only the final available diver slot is consumed in buffer order."""
    environment, params = gymnax.make("Seaquest-MinAtar")
    _, state = environment.reset(jax.random.key(0), params)
    divers = state.divers.at[0].set(jnp.array([5, 0, 1, 5]))
    divers = divers.at[1].set(jnp.array([5, 0, 1, 5]))
    state = state.replace(diver_count=jnp.array(5), divers=divers, divers_count=2)

    next_state = environment._step_divers(state, params)

    assert int(next_state.diver_count) == 6
    assert int(next_state.divers_count) == 1


def test_seaquest_full_rescue_matches_reference_diver_counter():
    """The upstream full-rescue transition leaves its historical -1 counter."""
    environment, params = gymnax.make("Seaquest-MinAtar")
    _, state = environment.reset(jax.random.key(0), params)
    state = state.replace(
        diver_count=jnp.array(6), surface=jnp.array(False), oxygen=jnp.array(180)
    )

    next_state, reward, done = environment._step_timers(state, params)

    assert int(next_state.diver_count) == -1
    assert float(reward) == 9.0
    assert not bool(done)


def test_seaquest_bullet_collision_removes_exactly_one_enemy():
    """A friendly bullet scores once and removes the colliding fish."""
    environment, params = gymnax.make("Seaquest-MinAtar")
    _, state = environment.reset(jax.random.key(0), params)
    state = state.replace(
        f_bullets=state.f_bullets.at[0].set(jnp.array([4, 1, 1])),
        f_bullet_count=jnp.array(1),
        e_fish=state.e_fish.at[0].set(jnp.array([5, 1, 1, 5])),
        e_fish_count=jnp.array(1),
    )

    next_state, reward = environment._step_friendly_bullets(state)

    assert float(reward) == 1.0
    assert int(next_state.f_bullet_count) == 0
    assert int(next_state.e_fish_count) == 0


def test_seaquest_oxygen_depletion_terminates_the_frame():
    """Seaquest follows the reference's oxygen termination ordering."""
    environment, params = gymnax.make("Seaquest-MinAtar")
    _, state = environment.reset(jax.random.key(0), params)
    state = state.replace(oxygen=jnp.array(0), sub_y=jnp.array(1))

    _, _, done = environment._step_timers(state, params)

    assert bool(done)


def test_seaquest_matches_reference_before_random_spawns():
    """Deterministic early frames retain MinAtar's action and bullet semantics."""
    reference = minatar_environment.Environment("seaquest", sticky_action_prob=0.0)
    environment, params = gymnax.make("Seaquest-MinAtar")
    key = jax.random.key(3)
    reference.reset()
    _, state = environment.reset(key, params)

    for action in (3, 5, 0, 0, 1, 5, 0, 2, 4, 0):
        reference_reward, reference_done = reference.act(action)
        key, step_key = jax.random.split(key)
        observation, state, reward, done, _ = environment.step_env(
            step_key, state, jnp.array(action), params
        )

        assert np.array_equal(observation, reference.state())
        assert float(reward) == reference_reward
        assert bool(done) == reference_done

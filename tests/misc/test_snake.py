"""Deterministic Snake transitions and JAX integration regressions."""

import jax
import jax.numpy as jnp
import pytest

import gymnax
from gymnax.environments.misc.snake import EnvState


def make_state(env, cells, apple, direction, time=0):
    """Build a coherent snake from cells ordered tail to head."""
    positions = jnp.asarray(cells, dtype=jnp.int32)
    board = jnp.zeros((env.height, env.width), dtype=jnp.int32)
    board = board.at[positions[:, 0], positions[:, 1]].set(
        jnp.arange(1, len(cells) + 1, dtype=jnp.int32)
    )
    return EnvState(
        board=board,
        head=positions[-1],
        direction=jnp.int32(direction),
        apple=jnp.asarray(apple, dtype=jnp.int32),
        length=jnp.int32(len(cells)),
        alive=jnp.bool_(True),
        time=jnp.int32(time),
    )


def assert_reset_state(env, params, key, observation, state):
    """Public step resets with its reset key while preserving terminal metadata."""
    _, reset_key = jax.random.split(key)
    expected_observation, expected_state = env.reset(reset_key, params)
    assert jnp.array_equal(observation, expected_observation)
    for actual, expected in zip(
        jax.tree.leaves(state), jax.tree.leaves(expected_state), strict=True
    ):
        assert jnp.array_equal(actual, expected)


def test_snake_is_registered_as_misc():
    """Registration exposes the documented action and observation spaces."""
    env, params = gymnax.make("Snake-misc")
    assert env.name == "Snake-misc"
    assert env.num_actions == env.action_space(params).n == 4
    assert env.observation_space(params).shape == (10, 10, 3)


def test_snake_growth_on_apple():
    """Eating retains the tail, adds a head, and places an apple on a free cell."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    params = params.replace(reward_apple=2.5, reward_step=0.25)
    state = make_state(env, [(2, 0), (2, 1)], (2, 2), direction=1)

    observation, next_state, reward, terminated, truncated, info = env.step(
        jax.random.key(0), state, jnp.int32(1), params
    )

    assert not terminated and not truncated
    assert next_state.alive
    assert next_state.length == info["length"] == 3
    assert next_state.time == 1
    assert jnp.array_equal(next_state.head, jnp.array([2, 2]))
    assert jnp.array_equal(next_state.board[2], jnp.array([1, 2, 3]))
    assert next_state.board[tuple(next_state.apple)] == 0
    assert reward == params.reward_apple + params.reward_step
    assert observation.dtype == jnp.float32
    assert observation[..., 0].sum() == 2
    assert observation[2, 2, 1] == 1
    assert observation[..., 2].sum() == 1


@pytest.mark.parametrize(
    "cells, expected_head, expected_direction",
    [
        ([(1, 0), (1, 1)], [1, 2], 1),
        ([(1, 1)], [1, 0], 3),
    ],
)
def test_snake_reverse_action_depends_on_length(
    cells, expected_head, expected_direction
):
    """Reverse actions are ignored for a body and allowed for a single cell."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    state = make_state(env, cells, (0, 0), direction=1)

    _, next_state, reward, terminated, truncated, _ = env.step(
        jax.random.key(1), state, jnp.int32(3), params
    )

    assert not terminated and not truncated
    assert next_state.direction == expected_direction
    assert jnp.array_equal(next_state.head, jnp.array(expected_head))
    assert next_state.length == len(cells)
    assert reward == params.reward_step


def test_snake_legal_tail_entry():
    """Moving into a live tail cell succeeds because the tail vacates it."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    state = make_state(env, [(1, 0), (2, 0), (2, 1), (1, 1)], (0, 2), direction=0)
    assert state.board[1, 0] == 1

    _, next_state, reward, terminated, truncated, _ = env.step(
        jax.random.key(2), state, jnp.int32(3), params
    )

    assert not terminated and not truncated
    assert next_state.alive and next_state.length == 4
    assert jnp.array_equal(next_state.head, jnp.array([1, 0]))
    assert next_state.board[1, 0] == 4
    assert jnp.array_equal(
        jnp.sort(next_state.board.reshape(-1))[-4:], jnp.arange(1, 5)
    )
    assert reward == params.reward_step


@pytest.mark.parametrize("collision", ["wall", "self"])
def test_snake_collision_terminates_and_resets(collision):
    """Real wall/body collisions return death reward and auto-reset metadata."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    if collision == "wall":
        state = make_state(env, [(0, 1)], (2, 2), direction=0)
        final_head = (0, 1)
    else:
        cells = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1)]
        state = make_state(env, cells, (2, 2), direction=1)
        final_head = (1, 1)
    key = jax.random.key(3)

    observation, next_state, reward, terminated, truncated, info = env.step(
        key, state, jnp.int32(0), params
    )

    assert terminated and not truncated
    assert info["terminated"] and not info["truncated"]
    assert reward == params.reward_death + params.reward_step
    assert info["discount"] == 0
    assert info["final_observation"][(*final_head, 1)] == 1
    assert_reset_state(env, params, key, observation, next_state)
    assert next_state.alive and next_state.time == 0


def test_snake_full_board_completion():
    """Eating the last free cell fills the board and auto-resets with success."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    cells = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1)]
    state = make_state(env, cells, (2, 2), direction=1)
    key = jax.random.key(4)
    step_key, _ = jax.random.split(key)

    final_observation, final_state, reward, terminated, _ = env.step_env(
        step_key, state, jnp.int32(1), params
    )
    observation, next_state, _, public_terminated, truncated, info = env.step(
        key, state, jnp.int32(1), params
    )

    assert terminated and public_terminated and not truncated
    assert final_state.alive and final_state.length == 9
    assert jnp.array_equal(jnp.sort(final_state.board.reshape(-1)), jnp.arange(1, 10))
    assert reward == params.reward_apple + params.reward_step
    assert final_observation[..., 0].sum() == 8
    assert final_observation[2, 2, 1] == 1
    assert jnp.array_equal(info["final_observation"], final_observation)
    assert info["length"] == 9 and info["discount"] == 0
    assert_reset_state(env, params, key, observation, next_state)


@pytest.mark.parametrize("time, expected_truncated", [(8, False), (9, True)])
def test_snake_truncation_at_time_limit(time, expected_truncated):
    """Safe transitions truncate exactly at the horizon without termination."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    params = params.replace(max_steps_in_episode=10)
    state = make_state(env, [(1, 1)], (0, 0), direction=1, time=time)
    key = jax.random.key(5)

    observation, next_state, reward, terminated, truncated, info = env.step(
        key, state, jnp.int32(1), params
    )

    assert not terminated and not info["terminated"]
    assert truncated == expected_truncated
    assert info["truncated"] == expected_truncated
    assert reward == params.reward_step
    assert info["discount"] == 1
    assert info["final_observation"][1, 2, 1] == 1
    if expected_truncated:
        assert_reset_state(env, params, key, observation, next_state)
    else:
        assert next_state.time == time + 1
        assert jnp.array_equal(next_state.head, jnp.array([1, 2]))


@pytest.mark.parametrize("height, width", [(10, 5), (5, 10), (2, 2)])
def test_snake_rectangular_board(height, width):
    """Compiled batched reset respects shape, bounds, and free apple placement."""
    env, params = gymnax.make("Snake-misc", height=height, width=width)
    keys = jax.random.split(jax.random.key(6), 16)
    observations, states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)

    assert observations.shape == (16, height, width, 3)
    assert observations.dtype == jnp.float32
    assert jnp.all(states.head >= 0) and jnp.all(
        states.head < jnp.array([height, width])
    )
    assert jnp.all(states.apple >= 0)
    assert jnp.all(states.apple < jnp.array([height, width]))
    assert jnp.all(jnp.any(states.head != states.apple, axis=-1))
    assert jnp.all(states.board.sum(axis=(1, 2)) == 1)
    assert jnp.all(observations[..., 0].sum(axis=(1, 2)) == 0)
    assert jnp.all(observations[..., 1:].sum(axis=(1, 2)) == 1)


def test_snake_jitted_vmap_step_matches_scalar_transitions():
    """Mixed growth, collision, ordinary, and truncation lanes match scalar steps."""
    env, params = gymnax.make("Snake-misc", height=5, width=5)
    params = params.replace(max_steps_in_episode=10)
    collision_cells = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1)]
    states = [
        make_state(env, [(2, 0), (2, 1)], (2, 2), direction=1),
        make_state(env, collision_cells, (4, 4), direction=1),
        make_state(env, [(2, 1)], (0, 0), direction=1),
        make_state(env, [(2, 1)], (0, 0), direction=1, time=9),
    ]
    keys = jax.random.split(jax.random.key(7), len(states))
    actions = jnp.array([1, 0, 1, 1], dtype=jnp.int32)
    batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)

    actual = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))(
        keys, batch, actions, params
    )
    expected = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *(
            env.step(keys[i], state, actions[i], params)
            for i, state in enumerate(states)
        ),
    )

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.array_equal(actual_leaf, expected_leaf)
    _, next_states, rewards, terminated, truncated, info = actual
    assert jnp.array_equal(terminated, jnp.array([False, True, False, False]))
    assert jnp.array_equal(truncated, jnp.array([False, False, False, True]))
    assert jnp.array_equal(rewards, jnp.array([1.0, -1.0, 0.0, 0.0]))
    assert jnp.array_equal(next_states.time, jnp.array([1, 0, 1, 0]))
    assert jnp.array_equal(info["length"], jnp.array([3, 8, 1, 1]))


def test_snake_canonical_termination_hook_and_discount():
    """Legacy hooks delegate to natural termination, excluding time limits."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    state = make_state(env, [(1, 1)], (0, 0), direction=1)
    assert not env.is_terminated(state, params)
    assert not env.is_terminal(state, params)
    assert env.discount(state, params) == 1
    terminal_state = state.replace(alive=jnp.bool_(False))
    assert env.is_terminated(terminal_state, params)
    assert env.is_terminal(terminal_state, params)
    assert env.discount(terminal_state, params) == 0
    assert not env.is_terminated(
        state.replace(time=params.max_steps_in_episode), params
    )


@pytest.mark.parametrize("height, width", [(0, 2), (2, 0), (1, 1), (2, 1), (1, 5)])
def test_snake_minimum_board_dimensions(height, width):
    """Unsupported board dimensions fail at construction."""
    with pytest.raises(ValueError, match="at least 2x2"):
        gymnax.make("Snake-misc", height=height, width=width)


def test_snake_state_space_bounds():
    """State space uses the correct per-axis coordinate bounds."""
    env, params = gymnax.make("Snake-misc", height=10, width=5)
    state_space = env.state_space(params)
    for coordinate in ["head", "apple"]:
        assert jnp.array_equal(state_space.spaces[coordinate].high, jnp.array([9, 4]))
    assert state_space.spaces["board"].shape == (10, 5)
    assert state_space.spaces["direction"].n == 4


def test_snake_no_apple_resample_without_eating():
    """Safe movement preserves the apple, ages the body, and vacates the tail."""
    env, params = gymnax.make("Snake-misc", height=3, width=3)
    state = make_state(env, [(1, 0), (1, 1)], (0, 0), direction=1)

    _, next_state, reward, terminated, truncated, _ = env.step(
        jax.random.key(8), state, jnp.int32(1), params
    )

    assert not terminated and not truncated
    assert jnp.array_equal(next_state.apple, state.apple)
    assert jnp.array_equal(next_state.board[1], jnp.array([0, 1, 2]))
    assert next_state.length == state.length
    assert reward == params.reward_step

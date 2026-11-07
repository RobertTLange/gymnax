"""Sampling correctness and runtime regressions for batched Snake steps."""

import jax
import jax.numpy as jnp
import pytest

from gymnax.environments.misc.snake import EnvState, Snake


class SamplingCounterSnake(Snake):
    """Record runtime calls to the full-board apple sampler."""

    def __init__(self, height=5, width=5):
        super().__init__(height, width)
        self.samples = []

    def _sample_apple(self, key, board):
        jax.debug.callback(lambda: self.samples.append(True))
        return super()._sample_apple(key, board)


def moving_state():
    return EnvState(
        board=jnp.zeros((5, 5), dtype=jnp.int32).at[2, 2].set(1),
        head=jnp.array([2, 2], dtype=jnp.int32),
        apple=jnp.array([0, 0], dtype=jnp.int32),
        direction=jnp.int32(1),
        length=jnp.int32(1),
        alive=jnp.bool_(True),
        time=jnp.int32(0),
    )


def test_reset_does_not_scan_the_board_to_sample_an_apple():
    env = SamplingCounterSnake()
    jax.block_until_ready(env.reset(jax.random.key(0)))
    jax.effects_barrier()

    assert not env.samples


def test_batched_non_eating_steps_skip_full_board_sampling():
    env = SamplingCounterSnake()
    keys = jax.random.split(jax.random.key(1), 4)
    states = jax.tree.map(lambda leaf: jnp.stack([leaf] * 4), moving_state())
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    result = step(keys, states, jnp.ones(4, dtype=jnp.int32), env.default_params)
    jax.block_until_ready(result)
    jax.effects_barrier()

    assert not env.samples
    assert jnp.array_equal(result[1].apple, states.apple)


@pytest.mark.parametrize("batch_shape", [(4,), (2, 2)])
@pytest.mark.parametrize("mapped_argument", ["key", "state"])
def test_sampling_guard_handles_shared_arguments_and_nested_vmap(
    batch_shape, mapped_argument
):
    env = SamplingCounterSnake()
    key = jax.random.key(5)
    state = moving_state()
    axes = (0, None) if mapped_argument == "key" else (None, 0)

    def step(key, state):
        return env.step(key, state, jnp.int32(1))

    for _ in batch_shape:
        step = jax.vmap(step, in_axes=axes)
    if mapped_argument == "key":
        key = jax.random.split(key, batch_shape)
    else:
        state = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (*batch_shape, *leaf.shape)), state
        )

    result = jax.jit(step)(key, state)
    jax.block_until_ready(result)
    jax.effects_barrier()

    assert not env.samples
    assert jnp.all(result[1].head == jnp.array([2, 3]))


def test_board_completion_does_not_sample_an_unusable_apple():
    env = SamplingCounterSnake(height=2, width=2)
    state = moving_state().replace(
        board=jnp.array([[1, 0], [2, 3]], dtype=jnp.int32),
        head=jnp.array([1, 1], dtype=jnp.int32),
        apple=jnp.array([0, 1], dtype=jnp.int32),
        length=jnp.int32(3),
    )

    result = env.step(jax.random.key(2), state, jnp.int32(0))
    jax.block_until_ready(result)
    jax.effects_barrier()

    assert result[3]
    assert not env.samples


def test_vmapped_step_env_accepts_shared_python_scalar_state_fields():
    env = Snake(height=5, width=5)
    state = moving_state().replace(time=0)
    keys = jax.random.split(jax.random.key(6), 4)

    result = jax.vmap(lambda key: env.step_env(key, state, 1, env.default_params))(keys)

    assert jnp.all(result[1].time == 1)
    assert jnp.all(result[1].head == jnp.array([2, 3]))


def test_reset_apple_is_uniform_over_cells_other_than_the_head():
    env = Snake(height=2, width=3)
    keys = jax.random.split(jax.random.key(3), 4096)
    _, states = jax.jit(jax.vmap(env.reset))(keys)
    heads = states.head[:, 0] * env.width + states.head[:, 1]
    apples = states.apple[:, 0] * env.width + states.apple[:, 1]
    counts = jnp.bincount(heads * 6 + apples, length=36).reshape(6, 6)
    off_diagonal = counts[~jnp.eye(6, dtype=jnp.bool_)]

    assert jnp.all(heads != apples)
    assert jnp.all(off_diagonal > 4096 / 30 / 2)
    assert jnp.all(off_diagonal < 4096 / 30 * 2)


def test_apple_sampler_is_uniform_over_free_cells():
    env = Snake(height=2, width=3)
    board = jnp.array([[0, 1, 0], [0, 2, 0]], dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(4), 4096)
    apples = jax.jit(jax.vmap(env._sample_apple, in_axes=(0, None)))(keys, board)
    indices = apples[:, 0] * env.width + apples[:, 1]
    counts = jnp.bincount(indices, length=6)

    assert counts[1] == counts[4] == 0
    assert jnp.all(counts[jnp.array([0, 2, 3, 5])] > 768)
    assert jnp.all(counts[jnp.array([0, 2, 3, 5])] < 1280)

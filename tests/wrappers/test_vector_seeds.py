import chex
import jax
import jax.numpy as jnp
import pytest

import gymnax
from gymnax.wrappers.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper


@pytest.mark.parametrize("seeds", [[0], [3, 7], [11, 11, 5]])
def test_vector_seed_list_matches_individual_environments(seeds):
    env, params = gymnax.make("CartPole-v1")
    vector = GymnaxToVectorGymWrapper(env, len(seeds), params)
    individuals = [GymnaxToGymWrapper(env, params) for _ in seeds]

    observations, _ = vector.reset(seed=seeds)
    expected = [single.reset(seed=seed)[0] for single, seed in zip(individuals, seeds)]
    chex.assert_trees_all_close(observations, jnp.stack(expected))

    for _ in range(3):
        observations, rewards, terminated, truncated, _ = vector.step(
            jnp.zeros(len(seeds), dtype=jnp.int32)
        )
        transitions = [single.step(0) for single in individuals]
        for values, index in [
            (observations, 0),
            (rewards, 1),
            (terminated, 2),
            (truncated, 3),
        ]:
            chex.assert_trees_all_close(
                values, jnp.stack([t[index] for t in transitions])
            )
        expected_states = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *[single.env_state for single in individuals],
        )
        chex.assert_trees_all_close(vector.env_state, expected_states)

    repeated, _ = vector.reset(seed=seeds)
    expected = [single.reset(seed=seed)[0] for single, seed in zip(individuals, seeds)]
    chex.assert_trees_all_close(repeated, jnp.stack(expected))

    continued, _ = vector.reset()
    expected = [single.reset()[0] for single in individuals]
    chex.assert_trees_all_close(continued, jnp.stack(expected))


@pytest.mark.parametrize("seeds", [[], [1], [1, 2, 3]])
def test_vector_seed_list_requires_one_seed_per_environment(seeds):
    env, params = gymnax.make("CartPole-v1")
    vector = GymnaxToVectorGymWrapper(env, 2, params)
    original_keys = jax.random.key_data(vector.key)
    original_state = vector.env_state

    with pytest.raises(ValueError, match="num_envs"):
        vector.reset(seed=seeds)

    chex.assert_trees_all_equal(jax.random.key_data(vector.key), original_keys)
    chex.assert_trees_all_equal(vector.env_state, original_state)


@pytest.mark.parametrize("seed", [None, 0, 13])
def test_vector_scalar_seed_keeps_existing_key_sequence(seed):
    env, params = gymnax.make("CartPole-v1")
    vector = GymnaxToVectorGymWrapper(env, 3, params)
    initial_keys = jax.random.split(jax.random.key(seed or 0), 3)
    next_keys, reset_keys = jax.vmap(jax.random.split, out_axes=1)(initial_keys)
    expected_observations, expected_state = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, params
    )

    observations, _ = vector.reset(seed=seed)

    chex.assert_trees_all_close(observations, expected_observations)
    chex.assert_trees_all_close(vector.env_state, expected_state)
    chex.assert_trees_all_equal(
        jax.random.key_data(vector.key), jax.random.key_data(next_keys)
    )

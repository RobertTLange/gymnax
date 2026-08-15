"""FrozenLake behavior tests."""

import jax
import jax.numpy as jnp

import gymnax


def test_frozen_lake_is_registered_with_gymnasium_style_spaces():
    """FrozenLake exposes scalar discrete observations and four actions."""
    env, params = gymnax.make("FrozenLake-misc")
    observation, state = env.reset(jax.random.key(0), params)

    assert env.action_space(params).n == 4
    assert env.observation_space(params).n == 16
    assert observation == 0
    assert state.position == 0


def test_frozen_lake_deterministic_goal_terminates_with_reward():
    """A deterministic route reaches the goal and returns its terminal observation."""
    env, params = gymnax.make("FrozenLake-misc", is_slippery=False)
    key = jax.random.key(0)
    _, state = env.reset(key, params)

    for action in (1, 1, 2, 2, 1, 2):
        key, step_key = jax.random.split(key)
        observation, state, reward, terminated, truncated, info = env.step(
            step_key, state, jnp.array(action), params
        )

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["final_observation"] == 15
    assert observation == 0
    assert state.position == 0


def test_frozen_lake_custom_map_rewards_and_truncates():
    """Custom maps support reward tuning and preserve time-limit semantics."""
    env, params = gymnax.make(
        "FrozenLake-misc",
        desc=("SG",),
        is_slippery=False,
        reward_schedule=(3.0, -2.0, -1.0),
    )
    params = params.replace(max_steps_in_episode=1)
    _, state = env.reset(jax.random.key(0), params)
    _, _, reward, terminated, truncated, _ = env.step(
        jax.random.key(1), state, jnp.array(2), params
    )

    assert reward == 3.0
    assert terminated
    assert truncated


def test_frozen_lake_supports_certain_transitions_and_map_specific_limits():
    """A success rate of one is deterministic and 8x8 uses its 200-step limit."""
    env, params = gymnax.make("FrozenLake-misc", map_name="8x8", success_rate=1.0)
    _, state = env.reset(jax.random.key(0), params)
    _, next_state, _, _, _, info = env.step(
        jax.random.key(1), state, jnp.array(2), params
    )

    assert params.max_steps_in_episode == 200
    assert next_state.position == 1
    assert info["p"] == 1.0


def test_frozen_lake_slippery_transitions_choose_intended_or_perpendicular_moves():
    """Default slippery actions have the Gymnasium one-third distribution."""
    env, _ = gymnax.make("FrozenLake-misc")
    keys = jax.random.split(jax.random.key(0), 3_000)
    actions, probabilities = jax.vmap(lambda key: env._transition_action(key, 0))(keys)
    counts = jnp.bincount(actions, length=4) / len(actions)

    assert counts[2] == 0.0
    assert jnp.all(jnp.abs(counts[jnp.array((0, 1, 3))] - 1.0 / 3.0) < 0.05)
    assert jnp.all(probabilities == 1.0 / 3.0)


def test_frozen_lake_holes_terminate_with_the_configured_reward():
    """Falling into a hole is natural termination, not time-limit truncation."""
    env, params = gymnax.make(
        "FrozenLake-misc",
        desc=("SHG",),
        is_slippery=False,
        reward_schedule=(1.0, -2.0, 0.0),
    )
    _, state = env.reset(jax.random.key(0), params)
    _, _, reward, terminated, truncated, info = env.step(
        jax.random.key(1), state, jnp.array(2), params
    )

    assert reward == -2.0
    assert terminated
    assert not truncated
    assert info["final_observation"] == 1

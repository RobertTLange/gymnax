import jax
import jax.numpy as jnp
import pytest

import gymnax


def split_env_state(state):
    # stack into list of trees (length = number of timesteps)
    return [jax.tree.map(lambda x: x[i], state) for i in range(len(state.time))]


def split_state_dict(state: dict):
    keys = state.keys()
    arrays = state.values()
    return [{k: v for k, v in zip(keys, values)} for values in zip(*arrays)]


@pytest.mark.parametrize(
    "env_id",
    [
        "CartPole-v1",  # Allows early termination
        "Pendulum-v1",  # No early termination
    ],
)
def test_truncation(env_id):
    env, env_params = gymnax.make(env_id)
    key = jax.random.PRNGKey(42)

    _, state = env.reset(key)
    action = env.action_space(env_params).sample(key)

    def step_fn(state, _):
        next_obs, next_state, _, done, info = env.step(key, state, action, env_params)
        return next_state, (next_obs, next_state, done, info)

    _, (observations, states, dones, infos) = jax.lax.scan(
        f=step_fn, init=state, xs=None, length=env_params.max_steps_in_episode + 1
    )

    # Should have at least finished once due to truncation
    assert sum(dones) >= 1
    infos = split_state_dict(infos)
    states = split_env_state(states)
    for i, (obs, state, done, info) in enumerate(
        zip(observations, states, dones, infos)
    ):
        if i == 0:
            # Need to observe the step before, not possible at i=0.
            continue
        if states[i - 1].time == env_params.max_steps_in_episode - 1:
            # Should have truncated
            assert info["truncated"]
            assert not info["terminated"]
            assert done
            # Last obs from finished episode should be different from \\
            # first obs of new episode
            assert not jnp.array_equal(info["obs_st"], obs)
        elif done:
            assert info["terminated"]
            assert not info["truncated"]
            # Last obs from finished episode should be different from \\
            # first obs of new episode
            assert not jnp.array_equal(info["obs_st"], obs)
        else:
            assert not info["truncated"]
            assert not info["terminated"]
            assert jnp.array_equal(info["obs_st"], obs)

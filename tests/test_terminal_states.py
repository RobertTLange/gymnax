import jax
import jax.numpy as jnp
import pytest

import gymnax


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
        return next_state, (next_obs, done, info)

    _, (observations, dones, infos) = jax.lax.scan(
        f=step_fn, init=state, xs=None, length=env_params.max_steps_in_episode + 1
    )

    # Should have at least finished once due to truncation
    assert sum(dones) >= 1
    for i, (obs, done, info) in enumerate(zip(observations, dones, infos)):
        if state.time == env_params.max_steps_in_episode - 1:
            if i + 1 <= len(observations):
                # Should have truncated
                assert infos[i + 1]["truncated"]
                assert not infos[i + 1]["terminated"]
                assert dones[i + 1]
                # Last obs from finished episode should be different from \\
                # first obs of new episode
                assert not jnp.array_equal(infos[i + 1]["obs_st"], obs)
        else:
            if done:
                assert info["terminated"]
                assert not info["truncated"]
                # Last obs from finished episode should be different from \\
                # first obs of new episode
                assert not jnp.array_equal(infos[i + 1]["obs_st"], obs)

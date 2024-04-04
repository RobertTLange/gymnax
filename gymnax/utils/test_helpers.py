"""Helper functions for testing."""

from typing import Any
import jax
import jaxlib
import numpy as np
from gymnax.utils import state_translate


def assert_correct_state(env_gym, env_name: str, state_jax: Any, atol: float = 1e-4):
    """Check that numpy-based env state is same as JAX dict."""

    state_gym = state_translate.np_state_to_jax(env_gym, env_name)
    # print(state_gym)
    # Loop over keys and assert that individual entries are same/close
    for k in state_gym.keys():
        jax_value = getattr(state_jax, k)
        # print(k, jax_value, state_gym[k])
        if k not in ["time", "terminal"]:
            if type(jax_value) in [
                jax.Array,
                # jaxlib.xla_extension.Buffer,
                jaxlib.xla_extension.ArrayImpl,
                np.ndarray,
            ]:
                assert np.allclose(jax_value, state_gym[k], atol=atol)
            else:
                # print(k, state_gym[k], state_jax[k])
                # Exclude extra time and terminal state from assertion
                if type(state_gym[k]) in [
                    float,
                    np.float64,
                    jax.Array,
                    # jaxlib.xla_extension.Buffer,
                    np.ndarray,
                    jaxlib.xla_extension.ArrayImpl,
                ]:
                    np.allclose(state_gym[k], jax_value, atol=atol)
                else:
                    print(type(state_gym[k]), k)
                    assert state_gym[k] == jax_value


def assert_correct_transit(
    obs_gym,
    reward_gym,
    done_gym,
    obs_jax,
    reward_jax,
    done_jax,
    atol: float = 1e-4,
):
    """Check that obs, reward, done transition outputs are correct."""
    if not done_gym:
        assert np.allclose(obs_gym, obs_jax, atol=atol)
    assert np.allclose(reward_gym, reward_jax, atol=atol)
    assert np.all(done_gym == done_jax)


def minatar_action_map(action_jax: int, env_name: str):
    """Helper that maps gymnax MinAtar action to the numpy equivalent."""
    all_actions = ["n", "l", "u", "r", "d", "f"]
    if env_name == "Asterix-MinAtar":
        minimal_actions = ["n", "l", "u", "r", "d"]
    elif env_name == "Breakout-MinAtar":
        minimal_actions = ["n", "l", "r"]
    elif env_name == "Freeway-MinAtar":
        minimal_actions = ["n", "u", "d"]
    elif env_name == "Seaquest-MinAtar":
        minimal_actions = ["n", "l", "u", "r", "d", "f"]
    elif env_name == "SpaceInvaders-MinAtar":
        minimal_actions = ["n", "l", "r", "f"]
    else:
        raise ValueError(f"{env_name} not in implemented MinAtar environments.")
    action_idx = all_actions.index(minimal_actions[action_jax])
    return action_idx

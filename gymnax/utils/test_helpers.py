import jax
import numpy as np
from .state_translate import np_state_to_jax


def assert_correct_state(env_gym, env_name: str, state_jax: dict,
                         atol: float=1e-4):
    """ Check that numpy-based env state is same as JAX dict. """
    state_gym = np_state_to_jax(env_gym, env_name)
    # Loop over keys and assert that individual entries are same/close
    for k in state_gym.keys():
        if type(state_jax[k]) == jax.interpreters.xla._DeviceArray:
            assert np.allclose(state_jax[k], state_gym[k], atol=atol)
        else:
            # Exclude extra time and terminal state from assertion
            if k not in ["time", "terminal"]:
                assert state_gym[k] == state_jax[k]


def assert_correct_transit(obs_gym, reward_gym, done_gym,
                           obs_jax, reward_jax, done_jax,
                           atol: float=1e-4):
    """ Check that obs, reward, done transition outputs are correct. """
    assert np.allclose(obs_gym, obs_jax, atol=atol)
    assert reward_gym == reward_jax
    assert done_gym == done_jax

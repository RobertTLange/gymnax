import chex
import jax
import gymnax
from gymnax.wrappers import GymnaxToDmEnvWrapper


def test_dmenv_wrapper():
    """Wrap a Gymnax environment in dm_env style."""
    env, env_params = gymnax.make("CartPole-v1")
    wrapped_env = GymnaxToDmEnvWrapper(env)
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    action = jax.vmap(env.action_space(env_params).sample)(keys)
    o, env_state = jax.vmap(env.reset)(keys)
    o, new_env_state, r, d, info = jax.vmap(env.step)(keys, env_state, action)

    reset_fn = jax.vmap(wrapped_env.reset, in_axes=(0,))
    timesteps = reset_fn(keys)
    chex.assert_tree_all_equal_shapes(o, timesteps.observation)
    chex.assert_tree_all_equal_shapes(r, timesteps.reward)
    chex.assert_tree_all_equal_shapes(d, timesteps.discount)

    step_fn = jax.vmap(wrapped_env.step, in_axes=(0, 0, 0))
    new_timesteps = step_fn(keys, timesteps, action)

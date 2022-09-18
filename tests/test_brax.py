import chex
import jax
import gymnax


def test_brax_wrapper():
    """Wrap a Gymnax environment in brax. Use Brax's wrappers to handle vmap and episodes"""
    try:
        from brax.envs.wrappers import VmapWrapper, EpisodeWrapper, EvalWrapper
        from gymnax.environments.conversions.brax import GymnaxToBraxWrapper
    except ImportError:
        return
    env, env_params = gymnax.make("CartPole-v1")
    brax_env = GymnaxToBraxWrapper(env)
    wrapped_env = VmapWrapper(EpisodeWrapper(brax_env, 100, 1))
    B = 16
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    action = jax.vmap(env.action_space(env_params).sample)(keys)
    reset_fn = jax.jit(wrapped_env.reset)
    o, env_state = jax.vmap(env.reset)(keys)
    o, new_env_state, r, d, info = jax.vmap(env.step)(keys, env_state, action)
    state = reset_fn(keys)
    chex.assert_tree_all_equal_shapes(o, state.obs)
    chex.assert_tree_all_equal_shapes(r, state.reward)
    chex.assert_tree_all_equal_shapes(d, state.done)
    chex.assert_tree_all_equal_structs(new_env_state, state.qp)
    step_fn = jax.jit(wrapped_env.step)
    new_state = step_fn(state, action)

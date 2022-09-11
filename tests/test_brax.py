import brax
import jax
from brax.envs.wrappers import VmapWrapper, EpisodeWrapper, EvalWrapper
from gymnax.environments.conversions.brax import GymnaxToBraxWrapper
import gymnax

def test_brax_wrapper():
    env, env_params = gymnax.make('CartPole-v1')
    brax_env = GymnaxToBraxWrapper(env)
    wrapped_env = VmapWrapper(EpisodeWrapper(brax_env, 100, 1))
    B = 16
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    reset_fn = jax.jit(wrapped_env.reset)
    state = reset_fn(keys)
    step_fn = wrapped_env.step
    new_state = step_fn(state, jax.numpy.tile(env.action_space(env_params).sample(keys[0]), (B, 1)))


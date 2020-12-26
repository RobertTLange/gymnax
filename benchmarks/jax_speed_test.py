import time
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax_ffw_policy import init_ffw_policy, ffw_policy
from gymnax import make_env


def policy_step(state_input, tmp):
    """ lax.scan compatible step transition in jax env. """
    obs, state, policy_params, env_params = state_input
    action = ffw_policy(policy_params, obs)
    next_o, next_s, reward, done, _ = step(env_params, state, action)
    carry, y = [next_o.squeeze(), next_s.squeeze(),
                policy_params, env_params], [reward]
    return carry, y


def policy_rollout(rng_input, policy_params, env_params, num_steps):
    """ Rollout a pendulum episode with lax.scan. """
    obs, state = reset(rng_input)
    scan_out1, scan_out2 = jax.lax.scan(policy_step,
                                        [obs, state, policy_params, env_params],
                                        [jnp.zeros(num_steps)])
    return scan_out1, jnp.array(scan_out2)


network_rollouts = jit(vmap(policy_rollout, in_axes=(0, None, None, None),
                            out_axes=0), static_argnums=(3))


def run_speed_test_jax(rng, num_episodes=50, num_env_steps=200, num_evals=100):
    """ Evaluate the runtime of gymnax-based OpenAI environments. """
    rng, rng_input = jax.random.split(rng)
    network_params = init_ffw_policy(rng_input, sizes=[3, 64, 1])
    rollout_keys = jax.random.split(rng, num_episodes)
    out1, out2 = network_rollouts(rollout_keys, network_params,
                                  env_params, num_env_steps)

    times = []
    for e in range(num_evals):
        start_t = time.time()
        rng, rng_input = jax.random.split(rng)
        rollout_keys = jax.random.split(rng, num_episodes)
        out1, out2 = network_rollouts(rollout_keys, network_params,
                                      env_params, num_env_steps)
        out2.block_until_ready()
        times.append(time.time() - start_t)
    print(sum(times)/num_evals)
    return


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    reset, step, env_params = make_env("Pendulum-v0")
    run_speed_test_jax(rng, num_episodes=200, num_env_steps=200, num_evals=200)

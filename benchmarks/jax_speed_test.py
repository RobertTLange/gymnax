import time
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax_ffw_policy import init_ffw_policy, ffw_policy
import gymnax
import gym


def policy_step(state_input, tmp):
    """ lax.scan compatible step transition in jax env. """
    rng, obs, state, policy_params, env_params = state_input
    rng, rng_input = jax.random.split(rng)
    action = ffw_policy(policy_params, obs)
    next_o, next_s, reward, done, _ = step(rng_input, env_params,
                                           state, action)
    carry, y = [rng, next_o.squeeze(), next_s.squeeze(),
                policy_params, env_params], [reward]
    return carry, y


def policy_rollout(rng_input, policy_params, env_params, num_steps):
    """ Rollout a pendulum episode with lax.scan. """
    obs, state = reset(rng_input, env_params)
    scan_out1, scan_out2 = jax.lax.scan(policy_step,
                                        [rng_input, obs, state, policy_params, env_params],
                                        [jnp.zeros(num_steps)])
    return scan_out1, jnp.array(scan_out2)


network_rollouts = jit(vmap(policy_rollout, in_axes=(0, None, None, None),
                            out_axes=0), static_argnums=(3))


def run_speed_test_jax_episode(rng, input_dim,
                               num_episodes=50, num_env_steps=200,
                               num_evals=100):
    """ Evaluate runtime of gymnax-based OpenAI environments. - Episodes """
    rng, rng_input = jax.random.split(rng)
    network_params = init_ffw_policy(rng_input, sizes=[input_dim, 64, 1])
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


def run_speed_test_jax_step(rng, env_name, num_evals=100):
    """ Evaluate the runtime of gymnax-based OpenAI environments. - Step """
    times = []
    env = gym.make(env_name)

    # Call once outside of benchmark loop for jit-compilation
    action = jnp.array(env.action_space.sample())
    rng, key_reset, key_step = jax.random.split(rng, 3)
    obs, state = reset(key_reset, env_params)
    start_t = time.time()
    obs, state, reward, done, _ = step(key_step, env_params,
                                       state, action)

    for e in range(num_evals):
        action = env.action_space.sample()
        rng, key_reset, key_step = jax.random.split(rng, 3)
        obs, state = reset(key_reset, env_params)
        start_t = time.time()
        obs, state, reward, done, _ = step(key_step, env_params,
                                           state, action)
        obs.block_until_ready()
        times.append(time.time() - start_t)
    print(sum(times)/num_evals)
    return


if __name__ == "__main__":
    env_names = ["Pendulum-v0", "CartPole-v0"]
    env_input_dims = {"Pendulum-v0": 3,
                      "CartPole-v0": 4}
    for seed_id, env_name in enumerate(env_names):
        rng, reset, step, env_params = gymnax.make(env_name, seed_id)
        run_speed_test_jax_step(rng, env_name, num_evals=1000)
        run_speed_test_jax_episode(rng,
                                   input_dim=env_input_dims[env_name],
                                   num_episodes=1,
                                   num_env_steps=200,
                                   num_evals=100)

import time
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import jaxlib

import gymnax
from policies.jax_policies import init_ffw_policy


def policy_rollout(rng_input, policy_params, env_params, num_steps):
    """ Rollout a pendulum episode with lax.scan. """
    # Single step transition helper function
    def policy_step(state_input, tmp):
        """ lax.scan compatible step transition in jax env. """
        rng, obs, state, policy_params, env_params = state_input
        rng, rng_input, rng_policy = jax.random.split(rng, 3)
        action = policy(rng_policy, policy_params, obs)
        next_o, next_s, reward, done, _ = step(rng_input, env_params,
                                               state, action)
        carry, y = [rng, next_o.squeeze(), next_s.squeeze(),
                    policy_params, env_params], [reward]
        return carry, y

    obs, state = reset(rng_input, env_params)
    scan_out1, scan_out2 = jax.lax.scan(policy_step,
                                        [rng_input, obs, state, policy_params, env_params],
                                        [jnp.zeros(num_steps)])
    return scan_out1, jnp.array(scan_out2)


episode_rollouts = jit(vmap(policy_rollout, in_axes=(0, None, None, None),
                            out_axes=0), static_argnums=(3))


def speed_ffw_jax_episode(rng, input_dim, output_dim, num_episodes=50,
                          num_env_steps=200, num_evals=100):
    """ Eval runtime of gymnax-based OpenAI environments. - FFW Policy """
    # Initialize network and get episode rollout keys ready
    rng, rng_input = jax.random.split(rng)
    network_params = init_ffw_policy(rng_input, sizes=[input_dim, 64,
                                                       output_dim])
    rollout_keys = jax.random.split(rng, num_episodes)

    # Run first episode outside of timed loop - includes compilation
    out1, out2 = episode_rollouts(rollout_keys, network_params,
                                  env_params, num_env_steps)

    times = []
    # Loop over individual episodes and collect time estimate data
    for e in range(num_evals):
        start_t = time.time()
        rng, rng_input = jax.random.split(rng)
        rollout_keys = jax.random.split(rng, num_episodes)
        out1, out2 = episode_rollouts(rollout_keys, network_params,
                                      env_params, num_env_steps)
        out2.block_until_ready()
        times.append(time.time() - start_t)
    return np.array(times)


def speed_random_jax_episode(rng, output_range, num_episodes=50,
                             num_env_steps=200, num_evals=100):
    """ Eval runtime of gymnax-based OpenAI environments. - Random Policy """
    # Initialize network and get episode rollout keys ready
    rng, rng_input = jax.random.split(rng)
    rollout_keys = jax.random.split(rng, num_episodes)
    # Run first episode outside of timed loop - includes compilation
    out1, out2 = episode_rollouts(rollout_keys, output_range,
                                  env_params, num_env_steps)

    times = []
    # Loop over individual episodes and collect time estimate data
    for e in range(num_evals):
        start_t = time.time()
        rng, rng_input = jax.random.split(rng)
        rollout_keys = jax.random.split(rng, num_episodes)
        out1, out2 = episode_rollouts(rollout_keys, output_range,
                                      env_params, num_env_steps)
        out2.block_until_ready()
        times.append(time.time() - start_t)
    return np.array(times)


if __name__ == "__main__":
    env_names = ["Pendulum-v0", "CartPole-v0",
                 "MountainCar-v0", "Acrobot-v1"]
    env_dims = {"Pendulum-v0": {"input": 3, "output": 1, "discrete": 0,
                                "output_range": jnp.array([-1, 1])},
                "CartPole-v0": {"input": 4, "output": 2, "discrete": 1,
                                "output_range": jnp.array([0, 1])},
                "MountainCar-v0": {"input": 2, "output": 3, "discrete": 1,
                                   "output_range": jnp.array([0, 1, 2]).astype(int)},
                "Acrobot-v1": {"input": 6, "output": 3, "discrete": 1,
                               "output_range": jnp.array([0, 1, 2])}}
    num_evals = 100
    num_steps = 200
    num_batch_episodes = 20
    all_envs_results = []
    devices = jax.devices()
    if type(devices[0]) == jaxlib.xla_extension.CpuDevice:
        device = "cpu"
    elif type(devices[0]) == jaxlib.xla_extension.GpuDevice:
        device = "gpu"
    elif type(devices[0]) == jaxlib.xla_extension.TpuDevice:
        device = "tpu"

    if device == "gpu":
        num_batch_episodes = 2000

    for seed_id, env_name in enumerate(env_names):
        print(env_name)
        # Import environment from gymnax
        rng, reset, step, env_params = gymnax.make(env_name, seed_id)

        #======================================================================

        # Run simulations for random episode rollouts
        if env_dims[env_name]["discrete"]:
            from policies.jax_policies import random_discrete_policy as policy
        else:
            from policies.jax_policies import random_continuous_policy as policy

        # Rollout single episode
        ep_random_times = speed_random_jax_episode(rng,
                                   output_range=env_dims[env_name]["output_range"],
                                   num_episodes=1,
                                   num_env_steps=num_steps,
                                   num_evals=num_evals)
        # Rollout batch episodes
        batch_random_times = speed_random_jax_episode(rng,
                                   output_range=env_dims[env_name]["output_range"],
                                   num_episodes=num_batch_episodes,
                                   num_env_steps=num_steps,
                                   num_evals=num_evals)

        #======================================================================

        # Run simulations for FFW architecture
        if env_dims[env_name]["discrete"]:
            from policies.jax_policies import ffw_discrete_policy as policy
        else:
            from policies.jax_policies import ffw_continuous_policy as policy

        # Rollout single episode
        ep_ffw_times = speed_ffw_jax_episode(rng,
                                   input_dim=env_dims[env_name]["input"],
                                   output_dim=env_dims[env_name]["output"],
                                   num_episodes=1,
                                   num_env_steps=num_steps,
                                   num_evals=num_evals)
        # Rollout batch episodes
        batch_ffw_times = speed_ffw_jax_episode(rng,
                                   input_dim=env_dims[env_name]["input"],
                                   output_dim=env_dims[env_name]["output"],
                                   num_episodes=num_batch_episodes,
                                   num_env_steps=num_steps,
                                   num_evals=num_evals)

        print(f"Random-Episode: {np.mean(ep_random_times)*1000}, {np.std(ep_random_times)*1000}")
        print(f"Random-Batch: {np.mean(batch_random_times)*1000}, {np.std(batch_random_times)*1000}")
        print(f"FFW-Episode: {np.mean(ep_ffw_times)*1000}, {np.std(ep_ffw_times)*1000}")
        print(f"FFW-Batch: {np.mean(batch_ffw_times)*1000}, {np.std(batch_ffw_times)*1000}")
        print(40*"=")

        result_dict = {"env_name": env_name,
                       "jax": 1,
                       "device": device,
                       "ep_random_mean": np.mean(ep_random_times),
                       "ep_random_std": np.std(ep_random_times),
                       "batch_random_mean": np.mean(batch_random_times),
                       "batch_random_std": np.std(batch_random_times),
                       "ep_ffw_mean": np.mean(ep_ffw_times),
                       "ep_ffw_std": np.std(ep_ffw_times),
                       "batch_ffw_mean": np.mean(batch_ffw_times),
                       "batch_ffw_std": np.std(batch_ffw_times)}
        all_envs_results.append(result_dict)

    df_to_store = pd.DataFrame(all_envs_results)
    df_to_store.to_csv("gymnax_speed_" + device + ".csv")

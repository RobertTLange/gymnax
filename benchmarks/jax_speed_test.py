import time
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import jaxlib

import gymnax
from policies.jax_policies import init_ffw_policy


def policy_rollout(rng_input, policy_params, num_steps):
    """ Rollout a pendulum episode with lax.scan. """
    # Single step transition helper function
    def policy_step(state_input, tmp):
        """ lax.scan compatible step transition in jax env. """
        rng, obs, state, policy_params = state_input
        rng, rng_step, rng_policy = jax.random.split(rng, 3)
        action = policy(rng_policy, policy_params, obs)
        next_o, next_s, reward, done, _ = env.step(rng_step, state, action)
        carry, y = [rng, next_o.squeeze(), next_s,
                    policy_params], [reward]
        return carry, y

    obs, state = env.reset(rng_input)
    scan_out1, scan_out2 = jax.lax.scan(policy_step,
                                        [rng_input, obs, state,
                                         policy_params],
                                        [jnp.zeros(num_steps)])
    return scan_out1, jnp.array(scan_out2)


episode_rollouts = jit(vmap(policy_rollout, in_axes=(0, None, None),
                            out_axes=0), static_argnums=(2))


def speed_jax_episode(rng, input_dim, output_dim, num_episodes=50,
                      num_env_steps=200, num_evals=100, random=False):
    """ Eval runtime of gymnax-based OpenAI environments. - FFW Policy """
    # Initialize network and get episode rollout keys ready
    rng, rng_input = jax.random.split(rng)
    # Differentiate between random rollout and FFW policy Rollout
    if not random:
        network_params = init_ffw_policy(rng_input, sizes=[input_dim, 64,
                                                           output_dim])
    else:
        network_params = output_dim
    rollout_keys = jax.random.split(rng, num_episodes)

    # Run first episode outside of timed loop - includes compilation
    out1, out2 = episode_rollouts(rollout_keys, network_params, num_env_steps)

    times = []
    # Loop over individual episodes and collect time estimate data
    for e in range(num_evals):
        start_t = time.time()
        rng, rng_input = jax.random.split(rng)
        rollout_keys = jax.random.split(rng, num_episodes)
        out1, out2 = episode_rollouts(rollout_keys, network_params,
                                      num_env_steps)
        out2.block_until_ready()
        times.append(time.time() - start_t)
    return np.array(times)


if __name__ == "__main__":
    gym_env_dims = {"Pendulum-v0": {"input": 3, "output": 1, "discrete": 0,
                                    "output_range": jnp.array([-1, 1])},
                    "CartPole-v0": {"input": 4, "output": 2, "discrete": 1,
                                    "output_range": jnp.array([0, 1])},
                    "MountainCar-v0": {"input": 2, "output": 3, "discrete": 1,
                                       "output_range": jnp.array([0, 1, 2])},
                    "Acrobot-v1": {"input": 6, "output": 3, "discrete": 1,
                                   "output_range": jnp.array([0, 1, 2])}}

    bsuite_env_dims = {
                "Catch-bsuite": {"input": 50, "output": 3, "discrete": 1,
                                 "output_range": jnp.array([0, 1, 2])},
                "DeepSea-bsuite": {"input": 64, "output": 2, "discrete": 1,
                                   "output_range": jnp.array([0, 1])},
                "DiscountingChain-bsuite": {"input": 2, "output": 5, "discrete": 1,
                                            "output_range": jnp.array([0, 1, 2, 3, 4])},
                "MemoryChain-bsuite": {"input": 3, "output": 2, "discrete": 1,
                                       "output_range": jnp.array([0, 1])},
                "UmbrellaChain-bsuite": {"input": 3, "output": 2, "discrete": 1,
                                         "output_range": jnp.array([0, 1])},
                "MNISTBandit-bsuite": {"input": 784, "output": 10, "discrete": 1,
                                       "output_range": jnp.arange(10)},
                "SimpleBandit-bsuite": {"input": 1, "output": 11, "discrete": 1,
                                        "output_range": jnp.arange(11)},}

    minatar_env_dims = {
                "Asterix-MinAtar": {"input": 500, "output": 5, "discrete": 1,
                                    "output_range": jnp.arange(5)},
                "Breakout-MinAtar": {"input": 400, "output": 3, "discrete": 1,
                                    "output_range": jnp.arange(3)},
                "Freeway-MinAtar": {"input": 700, "output": 3, "discrete": 1,
                                    "output_range": jnp.arange(3)},
                "SpaveInvaders-MinAtar": {"input": 600, "output": 4, "discrete": 1,
                                       "output_range": jnp.arange(4)},}

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

    env_dims = bsuite_env_dims
    print(40*"=")
    for seed_id, env_name in enumerate(env_dims.keys()):
        print(env_name)
        # Import environment from gymnax
        rng, env = gymnax.make(env_name, seed_id)

        #======================================================================
        # Run simulations for random episode rollouts
        if env_dims[env_name]["discrete"]:
            from policies.jax_policies import random_discrete_policy as policy
        else:
            from policies.jax_policies import random_continuous_policy as policy

        # Rollout single episode
        ep_random_times = speed_jax_episode(
                               rng,
                               input_dim=None,
                               output_dim=env_dims[env_name]["output_range"],
                               num_episodes=1,
                               num_env_steps=num_steps,
                               num_evals=num_evals,
                               random=True)
        # Rollout batch episodes
        batch_random_times = speed_jax_episode(
                               rng,
                               input_dim=None,
                               output_dim=env_dims[env_name]["output_range"],
                               num_episodes=num_batch_episodes,
                               num_env_steps=num_steps,
                               num_evals=num_evals,
                               random=True
                               )

        #======================================================================
        # Run simulations for FFW architecture
        if env_dims[env_name]["discrete"]:
            from policies.jax_policies import ffw_discrete_policy as policy
        else:
            from policies.jax_policies import ffw_continuous_policy as policy

        # Rollout single episode
        ep_ffw_times = speed_jax_episode(rng,
                                   input_dim=env_dims[env_name]["input"],
                                   output_dim=env_dims[env_name]["output"],
                                   num_episodes=1,
                                   num_env_steps=num_steps,
                                   num_evals=num_evals,
                                   random=False)
        # Rollout batch episodes
        batch_ffw_times = speed_jax_episode(rng,
                                   input_dim=env_dims[env_name]["input"],
                                   output_dim=env_dims[env_name]["output"],
                                   num_episodes=num_batch_episodes,
                                   num_env_steps=num_steps,
                                   num_evals=num_evals,
                                   random=False)

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
    df_to_store.to_csv("results/jax_speed_" + device + ".csv")

import time
import torch
import numpy as np
import pandas as pd
from vec_env.parallel import make_parallel_env
from policies.torch_policies import MLP_Policy


def speed_random_gym_episode(env_name, num_episodes=10,
                             num_env_steps=200, num_evals=100):
    """ Evaluate the runtime of gymnax-based OpenAI environments. """
    times = []
    envs = make_parallel_env(env_name, seed=0,
                             n_rollout_threads=num_episodes)
    for e in range(num_evals):
        envs.reset()
        start_t = time.time()
        for i in range(num_steps):
            action = [envs.action_space.sample() for e
                      in range(num_episodes)]
            obs, reward, done, _ = envs.step(action)
        times.append(time.time() - start_t)
    return np.array(times)


def speed_torch_gym_episode(env_name, input_dim, output_dim,
                            discrete, num_episodes=10,
                            num_env_steps=200, num_evals=100):
    times = []
    envs = make_parallel_env(env_name, seed=0,
                             n_rollout_threads=num_episodes)
    policy_net = MLP_Policy(input_dim, output_dim, discrete).to(device)

    for e in range(num_evals):
        obs = envs.reset()
        start_t = time.time()
        for i in range(num_steps):
            action = (policy_net(torch.Tensor(obs).to(device)).data.cpu()
                      .numpy())
            if env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]:
                action = np.atleast_1d(action.squeeze())
            obs, reward, done, _ = envs.step(action)
        times.append(time.time() - start_t)
    return np.array(times)


if __name__ == "__main__":
    env_dims = {"Pendulum-v0": {"input": 3, "output": 1, "discrete": 0},
                "CartPole-v0": {"input": 4, "output": 2, "discrete": 1},
                "MountainCar-v0": {"input": 2, "output": 3, "discrete": 1},
                "Acrobot-v1": {"input": 6, "output": 3, "discrete": 1},
                "MountainCarContinuous-v0": {"input": 2, "output": 1, "discrete": 0}}

    num_evals = 100
    num_steps = 200
    num_batch_episodes = 5
    all_envs_results = []
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    device_to_store = "gpu" if torch.cuda.is_available() else "cpu"

    if device_to_store == "gpu":
        num_batch_episodes = 40

    print(40*"=")
    for seed_id, env_name in enumerate(env_dims.keys()):
        print(env_name)

        #======================================================================

        # Rollout single episode
        ep_random_times = speed_random_gym_episode(env_name, 1,
                                                   num_steps, num_evals)

        # Rollout batch episodes
        batch_random_times = speed_random_gym_episode(env_name, num_batch_episodes,
                                                      num_steps, num_evals)

        #======================================================================
        # Rollout single episode
        ep_ffw_times = speed_torch_gym_episode(env_name,
                                        input_dim=env_dims[env_name]["input"],
                                        output_dim=env_dims[env_name]["output"],
                                        discrete=env_dims[env_name]["discrete"],
                                        num_episodes=1,
                                        num_env_steps=num_steps,
                                        num_evals=num_evals)

        # Rollout batch episodes
        batch_ffw_times = speed_torch_gym_episode(env_name,
                                        input_dim=env_dims[env_name]["input"],
                                        output_dim=env_dims[env_name]["output"],
                                        discrete=env_dims[env_name]["discrete"],
                                        num_episodes=num_batch_episodes,
                                        num_env_steps=num_steps,
                                        num_evals=num_evals)

        print(f"Random-Episode: {np.mean(ep_random_times)*1000}, {np.std(ep_random_times)*1000}")
        print(f"Random-Batch: {np.mean(batch_random_times)*1000}, {np.std(batch_random_times)*1000}")
        print(f"FFW-Episode: {np.mean(ep_ffw_times)*1000}, {np.std(ep_ffw_times)*1000}")
        print(f"FFW-Batch: {np.mean(batch_ffw_times)*1000}, {np.std(batch_ffw_times)*1000}")
        print(40*"=")

        result_dict = {"env_name": env_name,
                       "jax": 0,
                       "device": device_to_store,
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
    df_to_store.to_csv("results/gym_speed_" + device_to_store + ".csv")

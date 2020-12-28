import time
import gym


def run_speed_test_gym_step(env_name, num_evals=100):
    """ Evaluate the runtime of gymnax-based OpenAI environments. """
    times = []
    for e in range(num_evals):
        env = gym.make(env_name)
        env.reset()
        action = env.action_space.sample()
        start_t = time.time()
        obs, reward, done, _ = env.step(action)
        times.append(time.time() - start_t)
    print(sum(times)/num_evals)
    return


if __name__ == "__main__":
    env_names = ["Pendulum-v0", "CartPole-v0"]
    env_input_dims = {"Pendulum-v0": 3,
                      "CartPole-v0": 4}
    for seed_id, env_name in enumerate(env_names):
        run_speed_test_gym_step(env_name, num_evals=1000)

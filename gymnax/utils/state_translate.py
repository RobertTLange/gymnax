
def np_state_to_jax(env, env_name: str="Pendulum-v0"):
    """ Helper that collects env state into dict for JAX `step`. """
    if env_name in ["Pendulum-v0", "CartPole-v0",
                    "MountainCar-v0", "MountainCarContinuous-v0",
                    "Acrobot-v1"]:
        state_gym_to_jax = control_np_to_jax(env, env_name)
    elif env_name in ["Catch-bsuite", "DeepSea-bsuite",
                      "DiscountingChain-bsuite", "MemoryChain-bsuite",
                      "UmbrellaChain-bsuite", "MNISTBandit-bsuite",
                      "SimpleBandit-bsuite"]:
        state_gym_to_jax = bsuite_np_to_jax(env, env_name)
    elif env_name in ["Asterix-MinAtar",]:
        state_gym_to_jax = minatar_np_to_jax(env, env_name)
    else:
        raise ValueError(f"{env_name} is not in set of implemented"
                         " environments.")
    return state_gym_to_jax


def control_np_to_jax(env, env_name: str="Pendulum-v0"):
    """ Collects env state of classic_control into dict for JAX `step`. """
    if env_name == "Pendulum-v0":
        state_gym_to_jax = {"theta": env.state[0],
                            "theta_dot": env.state[1],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "CartPole-v0":
        state_gym_to_jax = {"x": env.state[0],
                            "x_dot": env.state[1],
                            "theta": env.state[2],
                            "theta_dot": env.state[3],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "MountainCar-v0":
        state_gym_to_jax = {"position": env.state[0],
                            "velocity": env.state[1],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "MountainCarContinuous-v0":
        state_gym_to_jax = {"position": env.state[0],
                            "velocity": env.state[1],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "Acrobot-v1":
        state_gym_to_jax = {"joint_angle1": env.state[0],
                            "joint_angle2": env.state[1],
                            "velocity_1": env.state[2],
                            "velocity_2": env.state[3],
                            "time": 0,
                            "terminal": 0}
    return state_gym_to_jax


def bsuite_np_to_jax(env, env_name: str="Catch-bsuite"):
    """ Collects env state of bsuite into dict for JAX `step`. """
    if env_name == "Catch-bsuite":
        state_gym_to_jax = {"ball_x": env._ball_x,
                            "ball_y": env._ball_y,
                            "paddle_x": env._paddle_x,
                            "paddle_y": env._paddle_y,
                            "prev_done": env._reset_next_step,
                            "time": 0,
                            "terminal": 0}
    elif env_name == "DeepSea-bsuite":
        state_gym_to_jax = {"row": env._row,
                            "column": env._column,
                            "bad_episode": env._bad_episode,
                            "total_bad_episodes": env._total_bad_episodes,
                            "denoised_return": env._denoised_return,
                            "optimal_return": env._optimal_return,
                            "action_mapping": env._action_mapping,
                            "time": 0,
                            "terminal": 0}
    elif env_name == "DiscountingChain-bsuite":
        state_gym_to_jax = {"rewards": env._rewards,
                            "context": env._context,
                            "time": env._timestep,
                            "terminal": 0}
    elif env_name == "MemoryChain-bsuite":
        state_gym_to_jax = {"context": env._context,
                            "query": env._query,
                            "total_perfect": env._total_perfect,
                            "total_regret":env._total_regret,
                            "time": env._timestep,
                            "terminal": 0}
    elif env_name == "UmbrellaChain-bsuite":
        state_gym_to_jax = {"need_umbrella": env._need_umbrella,
                            "has_umbrella": env._has_umbrella,
                            "total_regret": env._total_regret,
                            "time": env._timestep,
                            "terminal": 0}
    elif env_name == "MNISTBandit-bsuite":
        state_gym_to_jax = {"correct_label": env._correct_label,
                            "regret": env._total_regret,
                            "time": 0,
                            "terminal": 0}
    elif env_name == "SimpleBandit-bsuite":
        state_gym_to_jax = {"rewards": env._rewards,
                            "total_regret": env._total_regret,
                            "time": 0,
                            "terminal": 0}
    return state_gym_to_jax


def bsuite_np_to_jax(env, env_name: str="Asterix-MinAtar"):
    """ Collects env state of MinAtar into dict for JAX `step`. """
    if env_name == "Asterix-MinAtar":
        state_gym_to_jax = {
                            "time": 0,
                            "terminal": 0}
    return state_gym_to_jax

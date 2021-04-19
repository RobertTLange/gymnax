def np_state_to_jax(env, env_name: str="Pendulum-v0"):
    """ Helper that collects env state into dict for JAX `step`. """
    if env_name in ["Pendulum-v0", "CartPole-v0",
                    "MountainCar-v0", "MountainCarContinuous-v0",
                    "Acrobot-v1"]:
        state_gym_to_jax = control_np_to_jax(env, env_name)
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


def assert_state_correct(env_gym, env_name: str, state_jax: dict):
    """ Check that numpy-based env state is same as JAX dict. """
    state_gym = np_state_to_jax(env, env_name)
    # Loop over keys and assert that individual entries are same/close
    raise NotImplementedError


def assert_transition_correct():
    raise NotImplementedError

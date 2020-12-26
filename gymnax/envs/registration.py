from gymnax.envs.classic_control import reset_pendulum, step_pendulum, params_pendulum


def make_env(env_id: str):
    if env_id == "Pendulum-v0":
        reset, step, env_params = reset_pendulum, step_pendulum, params_pendulum
    else:
        raise ValueError("Env ID is not in set of environments.")
    return reset, step, env_params

from gymnax.envs.classic_control import (reset_pendulum, step_pendulum,
                                         params_pendulum)
from gymnax.envs.classic_control import (reset_cartpole, step_cartpole,
                                         params_cartpole)


def make(env_id: str):
    if env_id == "Pendulum-v0":
        reset, step, env_params = reset_pendulum, step_pendulum, params_pendulum
    elif env_id == "CartPole-v0":
        reset, step, env_params = reset_cartpole, step_cartpole, params_cartpole
    else:
        raise ValueError("Env ID is not in set of environments.")
    return reset, step, env_params

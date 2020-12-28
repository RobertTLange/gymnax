import jax
from gymnax.envs.classic_control import (reset_pendulum, step_pendulum,
                                         params_pendulum)
from gymnax.envs.classic_control import (reset_cartpole, step_cartpole,
                                         params_cartpole)
from gymnax.envs.classic_control import (reset_mountain_car, step_mountain_car,
                                         params_mountain_car)


def make(env_id: str, seed_id: int = 0):
    """ A JAX-version of of OpenAI's infamous env.make(env_name)"""
    if env_id == "Pendulum-v0":
        reset, step, env_params = reset_pendulum, step_pendulum, params_pendulum
    elif env_id == "CartPole-v0":
        reset, step, env_params = reset_cartpole, step_cartpole, params_cartpole
    elif env_id == "MountainCar-v0":
        reset, step, env_params = (reset_mountain_car, step_mountain_car,
                                   params_mountain_car)
    else:
        raise ValueError("Env ID is not in set of defined environments.")

    # Create a jax PRNG key for random seed control
    rng = jax.random.PRNGKey(seed_id)
    return rng, reset, step, env_params

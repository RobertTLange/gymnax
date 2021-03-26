import jax

# =============================================================================
from gymnax.environments.classic_control import (reset_pendulum, step_pendulum,
                                                 params_pendulum)
from gymnax.environments.classic_control import (reset_cartpole, step_cartpole,
                                                 params_cartpole)
from gymnax.environments.classic_control import (reset_mountain_car,
                                                 step_mountain_car,
                                                 params_mountain_car)
from gymnax.environments.classic_control import (reset_continuous_mountain_car,
                                                 step_continuous_mountain_car,
                                                 params_continuous_mountain_car)
from gymnax.environments.classic_control import (reset_acrobot, step_acrobot,
                                                 params_acrobot)

# =============================================================================
from gymnax.environments.bsuite import reset_catch, step_catch, params_catch

# =============================================================================
from gymnax.environments.minatar import (reset_asterix, step_asterix,
                                         params_asterix)
from gymnax.environments.minatar import (reset_breakout, step_breakout,
                                         params_breakout)
from gymnax.environments.minatar import (reset_freeway, step_freeway,
                                         params_freeway)

# =============================================================================
from gymnax.environments.misc import reset_bandit, step_bandit, params_bandit


def make(env_id: str, seed_id: int = 0):
    """ A JAX-version of of OpenAI's infamous env.make(env_name)"""
    # 1. Classic OpenAI Control Tasks
    if env_id == "Pendulum-v0":
        reset, step, env_params = reset_pendulum, step_pendulum, params_pendulum
    elif env_id == "CartPole-v0":
        reset, step, env_params = reset_cartpole, step_cartpole, params_cartpole
    elif env_id == "MountainCar-v0":
        reset, step, env_params = (reset_mountain_car, step_mountain_car,
                                   params_mountain_car)
    elif env_id == "MountainCarContinuous-v0":
        reset, step, env_params = (reset_continuous_mountain_car,
                                   step_continuous_mountain_car,
                                   params_continuous_mountain_car)
    elif env_id == "Acrobot-v1":
        reset, step, env_params = reset_acrobot, step_acrobot, params_acrobot
    # 2. DeepMind's bsuite environments
    elif env_id == "Catch-bsuite":
        reset, step, env_params = reset_catch, step_catch, params_catch
    # 3. MinAtar Environments
    elif env_id == "Asterix-MinAtar":
        reset, step, env_params = reset_asterix, step_asterix, params_asterix
    elif env_id == "Breakout-MinAtar":
        reset, step, env_params = reset_breakout, step_breakout, params_breakout
    elif env_id == "Freeway-MinAtar":
        reset, step, env_params = reset_freeway, step_freeway, params_freeway
    # 4. Other standard/popular environments
    elif env_id == "Bandit-misc":
        reset, step, env_params = reset_bandit, step_bandit, params_bandit
    else:
        raise ValueError("Env ID is not in set of defined environments.")

    # Create a jax PRNG key for random seed control
    rng = jax.random.PRNGKey(seed_id)
    return rng, reset, step, env_params

import jax

# =============================================================================
from gymnax.environments.classic_control import (Pendulum,
                                                 CartPole,
                                                 MountainCar,
                                                 ContinuousMountainCar,
                                                 Acrobot)

# # =============================================================================
from gymnax.environments.bsuite import (Catch,
                                        DeepSea,
                                        DiscountingChain,
                                        MemoryChain,
                                        UmbrellaChain)

# # =============================================================================
# from gymnax.environments.minatar import (reset_asterix, step_asterix,
#                                          params_asterix)
# from gymnax.environments.minatar import (reset_breakout, step_breakout,
#                                          params_breakout)
# from gymnax.environments.minatar import (reset_freeway, step_freeway,
#                                          params_freeway)
# from gymnax.environments.minatar import (reset_seaquest, step_seaquest,
#                                          params_seaquest)
# from gymnax.environments.minatar import (reset_space_invaders,
#                                          step_space_invaders,
#                                          params_space_invaders)
#
# # =============================================================================
# from gymnax.environments.misc import reset_bandit, step_bandit, params_bandit


def make(env_id: str, seed_id: int = 0):
    """ A JAX-version of OpenAI's infamous env.make(env_name)"""
    # 1. Classic OpenAI Control Tasks
    if env_id == "Pendulum-v0":
        env = Pendulum()
    elif env_id == "CartPole-v0":
        env = CartPole()
    elif env_id == "MountainCar-v0":
        env = MountainCar()
    elif env_id == "MountainCarContinuous-v0":
        env = ContinuousMountainCar()
    elif env_id == "Acrobot-v1":
        env = Acrobot()

    # 2. DeepMind's bsuite environments
    elif env_id == "Catch-bsuite":
        env = Catch()
    elif env_id == "DeepSea-bsuite":
        env = DeepSea()
    elif env_id == "DiscountingChain-bsuite":
        env = DiscountingChain()
    elif env_id == "MemoryChain-bsuite":
        env = MemoryChain()
    elif env_id == "UmbrellaChain-bsuite":
        env = UmbrellaChain()

    # # 3. MinAtar Environments
    # elif env_id == "Asterix-MinAtar":
    #     reset, step, env_params = reset_asterix, step_asterix, params_asterix
    # elif env_id == "Breakout-MinAtar":
    #     reset, step, env_params = (reset_breakout, step_breakout,
    #                                params_breakout)
    # elif env_id == "Freeway-MinAtar":
    #     reset, step, env_params = reset_freeway, step_freeway, params_freeway
    # elif env_id == "Seaquest-MinAtar":
    #     reset, step, env_params = (reset_seaquest, step_seaquest,
    #                                params_seaquest)
    # elif env_id == "SpaceInvaders-MinAtar":
    #     reset, step, env_params = (reset_space_invaders,
    #                                step_space_invaders,
    #                                params_space_invaders)
    #
    # # 4. Other standard/popular environments
    # elif env_id == "Bandit-misc":
    #     reset, step, env_params = reset_bandit, step_bandit, params_bandit
    # else:
    #     raise ValueError("Env ID is not in set of defined environments.")

    # Create a jax PRNG key for random seed control
    rng = jax.random.PRNGKey(seed_id)
    return rng, env

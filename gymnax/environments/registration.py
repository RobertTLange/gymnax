import jax

# =============================================================================
from gymnax.environments.classic_control import (Pendulum,
                                                 CartPole,
                                                 MountainCar,
                                                 ContinuousMountainCar,
                                                 Acrobot)
# =============================================================================
from gymnax.environments.bsuite import (Catch,
                                        DeepSea,
                                        DiscountingChain,
                                        MemoryChain,
                                        UmbrellaChain,
                                        MNISTBandit,
                                        SimpleBandit)
# =============================================================================
from gymnax.environments.minatar import (MinAsterix,
                                         MinBreakout,
                                         MinFreeway,
                                         MinSeaquest,
                                         MinSpaceInvaders)
# =============================================================================
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
    elif env_id == "MNISTBandit-bsuite":
        env = MNISTBandit()
    elif env_id == "SimpleBandit-bsuite":
        env = SimpleBandit()

    # 3. MinAtar Environments
    elif env_id == "Asterix-MinAtar":
        env = MinAsterix()
    elif env_id == "Breakout-MinAtar":
        env = MinBreakout()
    elif env_id == "Freeway-MinAtar":
        env = MinFreeway()
    elif env_id == "Seaquest-MinAtar":
        env = MinSeaquest()
    elif env_id == "SpaceInvaders-MinAtar":
        env = MinSpaceInvaders()

    # # 4. Other standard/popular environments
    # elif env_id == "Bandit-misc":
    #     reset, step, env_params = reset_bandit, step_bandit, params_bandit
    # else:
    #     raise ValueError("Env ID is not in set of defined environments.")

    # Create a jax PRNG key for random seed control
    rng = jax.random.PRNGKey(seed_id)
    return rng, env

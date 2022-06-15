from .environments import (
    Pendulum,
    CartPole,
    MountainCar,
    ContinuousMountainCar,
    Acrobot,
    Catch,
    DeepSea,
    DiscountingChain,
    MemoryChain,
    UmbrellaChain,
    MNISTBandit,
    SimpleBandit,
    MinAsterix,
    MinBreakout,
    MinFreeway,
    MinSeaquest,
    MinSpaceInvaders,
    BernoulliBandit,
    GaussianBandit,
    FourRooms,
    MetaMaze,
    PointRobot,
)

# =============================================================================


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's infamous env.make(env_name)"""
    # 1. Classic OpenAI Control Tasks
    if env_id == "Pendulum-v1":
        env = Pendulum(**env_kwargs)
    elif env_id == "CartPole-v1":
        env = CartPole(**env_kwargs)
    elif env_id == "MountainCar-v0":
        env = MountainCar(**env_kwargs)
    elif env_id == "MountainCarContinuous-v0":
        env = ContinuousMountainCar(**env_kwargs)
    elif env_id == "Acrobot-v1":
        env = Acrobot(**env_kwargs)

    # 2. DeepMind's bsuite environments
    elif env_id == "Catch-bsuite":
        env = Catch(**env_kwargs)
    elif env_id == "DeepSea-bsuite":
        env = DeepSea(**env_kwargs)
    elif env_id == "DiscountingChain-bsuite":
        env = DiscountingChain(**env_kwargs)
    elif env_id == "MemoryChain-bsuite":
        env = MemoryChain(**env_kwargs)
    elif env_id == "UmbrellaChain-bsuite":
        env = UmbrellaChain(**env_kwargs)
    elif env_id == "MNISTBandit-bsuite":
        env = MNISTBandit(**env_kwargs)
    elif env_id == "SimpleBandit-bsuite":
        env = SimpleBandit(**env_kwargs)

    # 3. MinAtar Environments
    elif env_id == "Asterix-MinAtar":
        env = MinAsterix(**env_kwargs)
    elif env_id == "Breakout-MinAtar":
        env = MinBreakout(**env_kwargs)
    elif env_id == "Freeway-MinAtar":
        env = MinFreeway(**env_kwargs)
    elif env_id == "Seaquest-MinAtar":
        env = MinSeaquest(**env_kwargs)
    elif env_id == "SpaceInvaders-MinAtar":
        env = MinSpaceInvaders(**env_kwargs)

    # 4. Miscellanoues Environments
    elif env_id == "BernoulliBandit-misc":
        env = BernoulliBandit(**env_kwargs)
    elif env_id == "GaussianBandit-misc":
        env = GaussianBandit(**env_kwargs)
    elif env_id == "FourRooms-misc":
        env = FourRooms(**env_kwargs)
    elif env_id == "MetaMaze-misc":
        env = MetaMaze(**env_kwargs)
    elif env_id == "PointRobot-misc":
        env = PointRobot(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    # Create a jax PRNG key for random seed control
    return env, env.default_params

"""A JAX implementation of OpenAI's infamous env.make(env_name)."""

from gymnax.environments.bsuite import (
    bandit,
    catch,
    deep_sea,
    discounting_chain,
    memory_chain,
    mnist,
    umbrella_chain,
)
from gymnax.environments.classic_control import (
    acrobot,
    cartpole,
    continuous_mountain_car,
    mountain_car,
    pendulum,
)
from gymnax.environments.minatar import asterix, breakout, freeway, space_invaders
from gymnax.environments.misc import (
    bernoulli_bandit,
    gaussian_bandit,
    meta_maze,
    point_robot,
    pong,
    reacher,
    rooms,
    swimmer,
)

# =============================================================================


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's infamous env.make(env_name).


    Args:
      env_id: A string identifier for the environment.
      **env_kwargs: Keyword arguments to pass to the environment.


    Returns:
      A tuple of the environment and the default parameters.
    """
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered gymnax environments.")

    # 1. Classic OpenAI Control Tasks
    if env_id == "Pendulum-v1":
        env = pendulum.Pendulum(**env_kwargs)
    elif env_id == "CartPole-v1":
        env = cartpole.CartPole(**env_kwargs)
    elif env_id == "MountainCar-v0":
        env = mountain_car.MountainCar(**env_kwargs)
    elif env_id == "MountainCarContinuous-v0":
        env = continuous_mountain_car.ContinuousMountainCar(**env_kwargs)
    elif env_id == "Acrobot-v1":
        env = acrobot.Acrobot(**env_kwargs)

    # 2. DeepMind's bsuite environments
    elif env_id == "Catch-bsuite":
        env = catch.Catch(**env_kwargs)
    elif env_id == "DeepSea-bsuite":
        env = deep_sea.DeepSea(**env_kwargs)
    elif env_id == "DiscountingChain-bsuite":
        env = discounting_chain.DiscountingChain(**env_kwargs)
    elif env_id == "MemoryChain-bsuite":
        env = memory_chain.MemoryChain(**env_kwargs)
    elif env_id == "UmbrellaChain-bsuite":
        env = umbrella_chain.UmbrellaChain(**env_kwargs)
    elif env_id == "MNISTBandit-bsuite":
        env = mnist.MNISTBandit(**env_kwargs)
    elif env_id == "SimpleBandit-bsuite":
        env = bandit.SimpleBandit(**env_kwargs)

    # 3. MinAtar Environments
    elif env_id == "Asterix-MinAtar":
        env = asterix.MinAsterix(**env_kwargs)
    elif env_id == "Breakout-MinAtar":
        env = breakout.MinBreakout(**env_kwargs)
    elif env_id == "Freeway-MinAtar":
        env = freeway.MinFreeway(**env_kwargs)
    elif env_id == "Seaquest-MinAtar":
        raise NotImplementedError("Seaquest is not yet supported.")
        # env = MinSeaquest(**env_kwargs)
    elif env_id == "SpaceInvaders-MinAtar":
        env = space_invaders.MinSpaceInvaders(**env_kwargs)

    # 4. Miscellanoues Environments
    elif env_id == "BernoulliBandit-misc":
        env = bernoulli_bandit.BernoulliBandit(**env_kwargs)
    elif env_id == "GaussianBandit-misc":
        env = gaussian_bandit.GaussianBandit(**env_kwargs)
    elif env_id == "FourRooms-misc":
        env = rooms.FourRooms(**env_kwargs)
    elif env_id == "MetaMaze-misc":
        env = meta_maze.MetaMaze(**env_kwargs)
    elif env_id == "PointRobot-misc":
        env = point_robot.PointRobot(**env_kwargs)
    elif env_id == "Reacher-misc":
        env = reacher.Reacher(**env_kwargs)
    elif env_id == "Swimmer-misc":
        env = swimmer.Swimmer(**env_kwargs)
    elif env_id == "Pong-misc":
        env = pong.Pong(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params


registered_envs = [
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    # "Seaquest-MinAtar",
    "SpaceInvaders-MinAtar",
    "Catch-bsuite",
    "DeepSea-bsuite",
    "MemoryChain-bsuite",
    "UmbrellaChain-bsuite",
    "DiscountingChain-bsuite",
    "MNISTBandit-bsuite",
    "SimpleBandit-bsuite",
    "FourRooms-misc",
    "MetaMaze-misc",
    "PointRobot-misc",
    "BernoulliBandit-misc",
    "GaussianBandit-misc",
    "Reacher-misc",
    "Swimmer-misc",
    "Pong-misc",
]

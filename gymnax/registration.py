"""Creation and runtime registration of Gymnax environments."""

from collections.abc import Callable
from typing import Any

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
from gymnax.environments.minatar import (
    asterix,
    breakout,
    freeway,
    seaquest,
    space_invaders,
)
from gymnax.environments.misc import (
    bernoulli_bandit,
    frozen_lake,
    gaussian_bandit,
    meta_maze,
    point_robot,
    pong,
    reacher,
    rooms,
    swimmer,
)

EnvironmentFactory = Callable[..., Any]

_BUILTIN_ENV_IDS = [
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "Seaquest-MinAtar",
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
    "FrozenLake-misc",
]

_builtins: dict[str, EnvironmentFactory] = {
    "Pendulum-v1": pendulum.Pendulum,
    "CartPole-v1": cartpole.CartPole,
    "MountainCar-v0": mountain_car.MountainCar,
    "MountainCarContinuous-v0": continuous_mountain_car.ContinuousMountainCar,
    "Acrobot-v1": acrobot.Acrobot,
    "Catch-bsuite": catch.Catch,
    "DeepSea-bsuite": deep_sea.DeepSea,
    "DiscountingChain-bsuite": discounting_chain.DiscountingChain,
    "MemoryChain-bsuite": memory_chain.MemoryChain,
    "UmbrellaChain-bsuite": umbrella_chain.UmbrellaChain,
    "MNISTBandit-bsuite": mnist.MNISTBandit,
    "SimpleBandit-bsuite": bandit.SimpleBandit,
    "Asterix-MinAtar": asterix.MinAsterix,
    "Breakout-MinAtar": breakout.MinBreakout,
    "Freeway-MinAtar": freeway.MinFreeway,
    "Seaquest-MinAtar": seaquest.MinSeaquest,
    "SpaceInvaders-MinAtar": space_invaders.MinSpaceInvaders,
    "BernoulliBandit-misc": bernoulli_bandit.BernoulliBandit,
    "GaussianBandit-misc": gaussian_bandit.GaussianBandit,
    "FourRooms-misc": rooms.FourRooms,
    "MetaMaze-misc": meta_maze.MetaMaze,
    "PointRobot-misc": point_robot.PointRobot,
    "Reacher-misc": reacher.Reacher,
    "Swimmer-misc": swimmer.Swimmer,
    "Pong-misc": pong.Pong,
    "FrozenLake-misc": frozen_lake.FrozenLake,
}
"""Factories for built-in and runtime-registered environments."""

_registry: dict[str, EnvironmentFactory] = {
    env_id: _builtins[env_id] for env_id in _BUILTIN_ENV_IDS
}

registered_envs = list(_registry)
"""Public environment IDs, updated in place by :func:`register`."""


def register(env_id: str, factory: EnvironmentFactory) -> None:
    """Register a factory used by :func:`make` to construct an environment.

    Args:
        env_id: Unique public ID for the environment.
        factory: Callable that accepts ``make`` keyword arguments and returns a
            new environment instance.

    Raises:
        TypeError: If the ID or factory has an invalid type.
        ValueError: If the ID is empty or already registered.
    """
    if not isinstance(env_id, str):
        raise TypeError("Environment ID must be a string")
    if not env_id:
        raise ValueError("Environment ID must not be empty")
    if not callable(factory):
        raise TypeError("Environment factory must be callable")
    if env_id in _registry:
        raise ValueError(f"{env_id} is already registered")

    _registry[env_id] = factory
    registered_envs.append(env_id)


def make(env_id: str, **env_kwargs):
    """Construct a registered environment and return it with default params."""
    try:
        factory = _registry[env_id]
    except KeyError as error:
        message = f"{env_id} is not in registered gymnax environments."
        raise ValueError(message) from error

    env = factory(**env_kwargs)
    required_attributes = ("default_params", "reset", "step")
    if any(not hasattr(env, attribute) for attribute in required_attributes):
        raise TypeError("Environment factory must return a Gymnax Environment")
    return env, env.default_params

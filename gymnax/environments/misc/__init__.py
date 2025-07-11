"""Miscellaneous environments in JAX.

This module provides JAX implementations of miscellaneous environments, including:
- BernoulliBandit
- GaussianBandit
- FourRooms
- FrozenLake
- MetaMaze
- PointRobot
- Reacher
- Swimmer
- Pong
"""

from .bernoulli_bandit import BernoulliBandit
from .gaussian_bandit import GaussianBandit
from .meta_maze import MetaMaze
from .point_robot import PointRobot
from .pong import Pong
from .reacher import Reacher
from .rooms import FourRooms
from .swimmer import Swimmer
from .frozen_lake import FrozenLake

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "FourRooms",
    "MetaMaze",
    "PointRobot",
    "Reacher",
    "Swimmer",
    "Pong",
    "FrozenLake",
]

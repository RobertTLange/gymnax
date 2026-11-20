"""Miscellaneous environments in JAX.

This package provides JAX implementations of the following environments:

- `BernoulliBandit`
- `GaussianBandit`
- `FourRooms`
- `FrozenLake`
- `MetaMaze`
- `PointRobot`
- `Reacher`
- `Snake`
- `Swimmer`
- `Pong`
"""

from .bernoulli_bandit import BernoulliBandit
from .frozen_lake import FrozenLake
from .gaussian_bandit import GaussianBandit
from .meta_maze import MetaMaze
from .point_robot import PointRobot
from .pong import Pong
from .reacher import Reacher
from .rooms import FourRooms
from .snake import Snake
from .swimmer import Swimmer

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "FourRooms",
    "FrozenLake",
    "MetaMaze",
    "PointRobot",
    "Reacher",
    "Snake",
    "Swimmer",
    "Pong",
]

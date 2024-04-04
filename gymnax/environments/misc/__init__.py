"""Miscellaneous environments."""

from gymnax.environments.misc import bernoulli_bandit
from gymnax.environments.misc import gaussian_bandit
from gymnax.environments.misc import meta_maze
from gymnax.environments.misc import point_robot
from gymnax.environments.misc import pong
from gymnax.environments.misc import reacher
from gymnax.environments.misc import rooms
from gymnax.environments.misc import swimmer


BernoulliBandit = bernoulli_bandit.BernoulliBandit
GaussianBandit = gaussian_bandit.GaussianBandit
MetaMaze = meta_maze.MetaMaze
PointRobot = point_robot.PointRobot
Pong = pong.Pong
Reacher = reacher.Reacher
Swimmer = swimmer.Swimmer
FourRooms = rooms.FourRooms


__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "FourRooms",
    "MetaMaze",
    "PointRobot",
    "Reacher",
    "Swimmer",
    "Pong",
]

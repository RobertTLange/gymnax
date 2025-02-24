"""Miscellaneous environments."""

from gymnax.environments.misc import (bernoulli_bandit, gaussian_bandit,
                                      meta_maze, point_robot, pong, reacher,
                                      rooms, swimmer)

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

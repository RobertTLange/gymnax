"""Gymnax environments."""

from gymnax.environments import bsuite
from gymnax.environments import classic_control
from gymnax.environments import environment
from gymnax.environments import minatar
from gymnax.environments import misc


Catch = bsuite.Catch
DeepSea = bsuite.DeepSea
DiscountingChain = bsuite.DiscountingChain
MemoryChain = bsuite.MemoryChain
MNISTBandit = bsuite.MNISTBandit
SimpleBandit = bsuite.SimpleBandit
UmbrellaChain = bsuite.UmbrellaChain
Acrobot = classic_control.Acrobot
CartPole = classic_control.CartPole
ContinuousMountainCar = classic_control.ContinuousMountainCar
MountainCar = classic_control.MountainCar
Pendulum = classic_control.Pendulum
EnvState = environment.EnvState
EnvParams = environment.EnvParams
MinAsterix = minatar.MinAsterix
MinBreakout = minatar.MinBreakout
MinFreeway = minatar.MinFreeway
# MinSeaquest = minatar.MinSeaquest
MinSpaceInvaders = minatar.MinSpaceInvaders
BernoulliBandit = misc.BernoulliBandit
FourRooms = misc.FourRooms
GaussianBandit = misc.GaussianBandit
MetaMaze = misc.MetaMaze
PointRobot = misc.PointRobot
Pong = misc.Pong
Reacher = misc.Reacher
Swimmer = misc.Swimmer


__all__ = [
    "EnvParams",
    "EnvState",
    "Pendulum",
    "CartPole",
    "MountainCar",
    "ContinuousMountainCar",
    "Acrobot",
    "Catch",
    "DeepSea",
    "DiscountingChain",
    "MemoryChain",
    "UmbrellaChain",
    "MNISTBandit",
    "SimpleBandit",
    "MinAsterix",
    "MinBreakout",
    "MinFreeway",
    # "MinSeaquest",
    "MinSpaceInvaders",
    "BernoulliBandit",
    "GaussianBandit",
    "FourRooms",
    "MetaMaze",
    "PointRobot",
    "Reacher",
    "Swimmer",
    "Pong",
]

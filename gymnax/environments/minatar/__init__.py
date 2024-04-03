"""Minatar environments."""

from gymnax.environments.minatar import asterix
from gymnax.environments.minatar import breakout
from gymnax.environments.minatar import freeway
from gymnax.environments.minatar import space_invaders

# from gymnax.environments.minatar import seaquest


MinAsterix = asterix.MinAsterix
MinBreakout = breakout.MinBreakout
MinFreeway = freeway.MinFreeway
# MinSeaquest = seaquest.MinSeaquest
MinSpaceInvaders = space_invaders.MinSpaceInvaders


__all__ = [
    "MinAsterix",
    "MinBreakout",
    "MinFreeway",
    # "MinSeaquest",
    "MinSpaceInvaders",
]

"""Minatar environments."""

from gymnax.environments.minatar import (asterix, breakout, freeway,
                                         space_invaders)

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

"""JAX implementations of the MinAtar environments.

This package provides the following environments (environment name in parentheses):

- `MinAsterix` (`Asterix-MinAtar`)
- `MinBreakout` (`Breakout-MinAtar`)
- `MinFreeway` (`Freeway-MinAtar`)
- `MinSeaquest` (`Seaquest-MinAtar`)
- `MinSpaceInvaders` (`SpaceInvaders-MinAtar`)
"""

from .asterix import MinAsterix
from .breakout import MinBreakout
from .freeway import MinFreeway
from .seaquest import MinSeaquest
from .space_invaders import MinSpaceInvaders

__all__ = [
    "MinAsterix",
    "MinBreakout",
    "MinFreeway",
    "MinSeaquest",
    "MinSpaceInvaders",
]

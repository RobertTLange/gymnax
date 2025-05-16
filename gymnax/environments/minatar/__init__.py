"""Minatar environments in JAX.

This module provides JAX implementations of MinAtar environments, including:
- Asterix
- Breakout
- Freeway
- Space Invaders
"""

from .asterix import MinAsterix
from .breakout import MinBreakout
from .freeway import MinFreeway
from .space_invaders import MinSpaceInvaders

__all__ = [
    "MinAsterix",
    "MinBreakout",
    "MinFreeway",
    "MinSpaceInvaders",
]

"""Bsuite environments in JAX.

This module provides JAX implementations of bsuite environments, including:
- Catch
- DeepSea
- DiscountingChain
- MemoryChain
- UmbrellaChain
- MNISTBandit
- SimpleBandit
"""

from .bandit import SimpleBandit
from .catch import Catch
from .deep_sea import DeepSea
from .discounting_chain import DiscountingChain
from .memory_chain import MemoryChain
from .mnist import MNISTBandit
from .umbrella_chain import UmbrellaChain

__all__ = [
    "Catch",
    "DeepSea",
    "DiscountingChain",
    "MemoryChain",
    "UmbrellaChain",
    "MNISTBandit",
    "SimpleBandit",
]

"""Bsuite environments in JAX.

JAX ports of the bsuite environments:

- `Catch`: `Catch-bsuite`
- `DeepSea`: `DeepSea-bsuite`
- `DiscountingChain`: `DiscountingChain-bsuite`
- `MemoryChain`: `MemoryChain-bsuite`
- `UmbrellaChain`: `UmbrellaChain-bsuite`
- `MNISTBandit`: `MNISTBandit-bsuite`
- `SimpleBandit`: `SimpleBandit-bsuite`
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

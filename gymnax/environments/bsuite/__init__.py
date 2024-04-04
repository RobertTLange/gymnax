"""Bsuite environments."""

from gymnax.environments.bsuite import bandit
from gymnax.environments.bsuite import catch
from gymnax.environments.bsuite import deep_sea
from gymnax.environments.bsuite import discounting_chain
from gymnax.environments.bsuite import memory_chain
from gymnax.environments.bsuite import mnist
from gymnax.environments.bsuite import umbrella_chain


SimpleBandit = bandit.SimpleBandit
Catch = catch.Catch
DeepSea = deep_sea.DeepSea
DiscountingChain = discounting_chain.DiscountingChain
MemoryChain = memory_chain.MemoryChain
MNISTBandit = mnist.MNISTBandit
UmbrellaChain = umbrella_chain.UmbrellaChain


__all__ = [
    "Catch",
    "DeepSea",
    "DiscountingChain",
    "MemoryChain",
    "UmbrellaChain",
    "MNISTBandit",
    "SimpleBandit",
]

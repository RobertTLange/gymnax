"""Bsuite environments."""

from gymnax.environments.bsuite import (bandit, catch, deep_sea,
                                        discounting_chain, memory_chain, mnist,
                                        umbrella_chain)

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

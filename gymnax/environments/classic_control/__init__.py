"""Classic control environments in JAX.

This module provides JAX implementations of classic control environments
from OpenAI Gym, including:
- Acrobot
- CartPole
- MountainCar
- ContinuousMountainCar
- Pendulum
"""

from .acrobot import Acrobot
from .cartpole import CartPole
from .continuous_mountain_car import ContinuousMountainCar
from .mountain_car import MountainCar
from .pendulum import Pendulum

__all__ = [
    "Acrobot",
    "CartPole",
    "MountainCar",
    "ContinuousMountainCar",
    "Pendulum",
]

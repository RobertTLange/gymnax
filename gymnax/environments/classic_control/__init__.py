"""Classic control environments in JAX.

JAX ports of the Gymnasium classic control environments:

- `Acrobot`: `Acrobot-v1`
- `CartPole`: `CartPole-v1`
- `MountainCar`: `MountainCar-v0`
- `ContinuousMountainCar`: `MountainCarContinuous-v0`
- `Pendulum`: `Pendulum-v1`
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

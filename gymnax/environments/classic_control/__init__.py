"""Classic control environments."""

from gymnax.environments.classic_control import (acrobot, cartpole,
                                                 continuous_mountain_car,
                                                 mountain_car, pendulum)

Acrobot = acrobot.Acrobot
CartPole = cartpole.CartPole
ContinuousMountainCar = continuous_mountain_car.ContinuousMountainCar
MountainCar = mountain_car.MountainCar
Pendulum = pendulum.Pendulum


__all__ = [
    "Pendulum",
    "CartPole",
    "MountainCar",
    "ContinuousMountainCar",
    "Acrobot",
]

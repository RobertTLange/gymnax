"""Gymnax: A library for creating and registering Gym environments."""

from gymnax import environments
from gymnax import registration

EnvParams = environments.EnvParams
EnvState = environments.EnvState
make = registration.make
registered_envs = registration.registered_envs


__all__ = ["make", "registered_envs", "EnvState", "EnvParams"]

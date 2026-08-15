"""Gymnax: A library for creating and registering Gym environments."""

from gymnax import environments, registration

EnvParams = environments.EnvParams
EnvState = environments.EnvState
make = registration.make
register = registration.register
registered_envs = registration.registered_envs


__all__ = ["make", "register", "registered_envs", "EnvState", "EnvParams"]

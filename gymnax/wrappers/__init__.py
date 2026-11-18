"""Wrappers that modify Gymnax environments or adapt them to other APIs."""

from gymnax.wrappers import compat, dm_env, gym, purerl

GymnaxToDmEnvWrapper = dm_env.GymnaxToDmEnvWrapper
GymnaxToGymWrapper = gym.GymnaxToGymWrapper
GymnaxToVectorGymWrapper = gym.GymnaxToVectorGymWrapper
LegacyStepAPIWrapper = compat.LegacyStepAPIWrapper
FlattenObservationWrapper = purerl.FlattenObservationWrapper
LogWrapper = purerl.LogWrapper
StickyActionWrapper = purerl.StickyActionWrapper


__all__ = [
    "GymnaxToDmEnvWrapper",
    "GymnaxToGymWrapper",
    "GymnaxToVectorGymWrapper",
    "LegacyStepAPIWrapper",
    "FlattenObservationWrapper",
    "LogWrapper",
    "StickyActionWrapper",
]

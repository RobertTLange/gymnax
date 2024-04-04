"""Wrappers for Gymnax environments to be compatible with Brax."""

from typing import Any, Dict, Optional, Union


import chex
from flax import struct
import jax
from gymnax.environments import environment

try:
    from brax import envs
except ImportError as exc:
    raise ModuleNotFoundError(
        "You need to install `brax` to use the brax wrapper."
    ) from exc


@struct.dataclass
class State:  # Lookalike for brax.envs.env.State.
    qp: environment.EnvState  # Brax QP is roughly equivalent to our EnvState
    obs: Any  # depends on environment
    reward: float
    done: bool
    metrics: Dict[str, Union[chex.Array, chex.Scalar]] = struct.field(
        default_factory=dict
    )
    info: Dict[str, Any] = struct.field(default_factory=dict)


class GymnaxToBraxWrapper(envs.Env):
    """Wrap Gymnax environment as Brax environment.

    Primarily useful for including obs, reward, and done as part of state.
    Compatible with all brax wrappers, but AutoResetWrapper is redundant
    since Gymnax environments already reset state.
    """

    def __init__(self, env: environment.Environment):
        super().__init__()
        self.env = env

    def reset(
        self, rng: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ):  # -> State:
        """Reset, return brax State. Save rng and params in info field for step."""
        if params is None:
            params = self.env.default_params
        obs, env_state = self.env.reset(rng, params)
        return State(
            env_state,
            obs,
            0.0,
            False,
            {},
            {"_rng": jax.random.split(rng)[0], "_env_params": params},
        )

    def step(
        self,
        state,  #: State,
        action,  #: Union[chex.Scalar, chex.Array],
        params=None,  #: Optional[environment.EnvParams] = None,
    ):  # -> State:
        """Step brax State. Update stored rng and params in info field."""
        rng, step_rng = jax.random.split(state.info["_rng"])
        if params is None:
            params = self.env.default_params
        state.info.update(_rng=rng, _env_params=params)
        o, env_state, r, d, _ = self.env.step(step_rng, state.qp, action, params)
        return state.replace(qp=env_state, obs=o, reward=r, done=d)

    def action_size(self) -> int:
        """DEFAULT size of action vector expected by step."""
        a_space = self.env.action_space(self.env.default_params)
        example_a = a_space.sample(jax.random.PRNGKey(0))
        return len(jax.tree_util.tree_flatten(example_a)[0])

    def observation_size(self) -> int:
        """DEFAULT size of observation vector expected by step."""
        o_space = self.env.observation_space(self.env.default_params)
        example_o = o_space.sample(jax.random.PRNGKey(0))
        return len(jax.tree_util.tree_flatten(example_o)[0])

    def backend(self) -> str:
        """Return backend of the environment."""
        return "jax"

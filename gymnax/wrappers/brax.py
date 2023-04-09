from typing import Any, Dict, Union, Optional

try:
    from brax.envs.env import Env
except ImportError:
    raise ImportError("You need to install `brax` to use the brax wrapper.")
import jax
import chex
from ..environment import Environment, EnvState, EnvParams
from flax.struct import dataclass, field


@dataclass
class State:  # Lookalike for brax.envs.env.State
    qp: EnvState  # Brax QP is roughly equivalent to our EnvState
    obs: Any  # depends on environment
    reward: float
    done: bool
    metrics: Dict[str, Union[chex.Array, chex.Scalar]] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


class GymnaxToBraxWrapper(Env):
    def __init__(self, env: Environment):
        """Wrap Gymnax environment as Brax environment

        Primarily useful for including obs, reward, and done as part of state.
        Compatible with all brax wrappers, but AutoResetWrapper is redundant since Gymnax environments
        already reset state.

        Args:
            env: Gymnax environment instance
        """
        super().__init__("")
        self.env = env

    def reset(self, rng: chex.PRNGKey, params: Optional[EnvParams] = None) -> State:
        """Reset, return brax State. Save rng and params in info field for step"""
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
        state: State,
        action: Union[chex.Scalar, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> State:
        """Step, return brax State. Update stored rng and params (if provided) in info field"""
        rng, step_rng = jax.random.split(state.info["_rng"])
        if params is None:
            params = self.env.default_params
        state.info.update(_rng=rng, _env_params=params)
        o, env_state, r, d, info = self.env.step(step_rng, state.qp, action, params)
        return state.replace(qp=env_state, obs=o, reward=r, done=d)

    def action_size(self) -> int:
        """DEFAULT size of action vector expected by step. Can't pass params to property"""
        a_space = self.env.action_space(self.env.default_params)
        example_a = a_space.sample(jax.random.PRNGKey(0))
        return len(jax.tree_util.tree_flatten(example_a)[0])

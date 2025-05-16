"""Wrappers for Gymnax environments to be compatible with Brax."""

import jax

from gymnax.environments import environment

try:
    from brax import envs
    from brax.envs import State
except ImportError as exc:
    raise ModuleNotFoundError(
        "You need to install `brax` to use the brax wrapper."
    ) from exc


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
        self, key: jax.Array, params: environment.EnvParams | None = None
    ):  # -> State:
        """Reset, return brax State. Save key and params in info field for step."""
        if params is None:
            params = self.env.default_params
        obs, env_state = self.env.reset(key, params)
        return State(
            pipeline_state=env_state,
            obs=obs,
            reward=jax.numpy.array(0.0),
            done=jax.numpy.array(False),
            metrics={},
            info={"_key": jax.random.split(key)[0], "_env_params": params},
        )

    def step(
        self,
        state,  #: State,
        action,  #: Union[jax.Array, jax.Array],
        params=None,  #: Optional[environment.EnvParams] = None,
    ):  # -> State:
        """Step brax State. Update stored key and params in info field."""
        key, step_key = jax.random.split(state.info["_key"])
        if params is None:
            params = self.env.default_params
        state.info.update(_key=key, _env_params=params)
        o, env_state, r, d, _ = self.env.step(
            step_key, state.pipeline_state, action, params
        )
        return state.replace(
            pipeline_state=env_state,
            obs=o,
            reward=jax.numpy.array(r),
            done=jax.numpy.array(d),
        )

    def action_size(self) -> int:
        """DEFAULT size of action vector expected by step."""
        a_space = self.env.action_space(self.env.default_params)
        example_a = a_space.sample(jax.random.key(0))
        return len(jax.tree.flatten(example_a)[0])

    def observation_size(self) -> int:
        """DEFAULT size of observation vector expected by step."""
        o_space = self.env.observation_space(self.env.default_params)
        example_o = o_space.sample(jax.random.key(0))
        return len(jax.tree.flatten(example_o)[0])

    def backend(self) -> str:
        """Return backend of the environment."""
        return "jax"

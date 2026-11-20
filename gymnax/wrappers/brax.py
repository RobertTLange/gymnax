"""Wrapper exposing Gymnax environments through the Brax API."""

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
    """Wrap a Gymnax environment as a Brax environment.

    Observation, reward and done flag become part of the Brax `State`, and the PRNG
    key and environment parameters are carried in `state.info`. Compatible with all
    Brax wrappers; `AutoResetWrapper` is redundant because Gymnax environments
    already reset automatically. Importing this module requires `brax`.

    Args:
        env: Gymnax environment to wrap.
    """

    def __init__(self, env: environment.Environment):
        super().__init__()
        self.env = env

    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ):  # -> State:
        """Reset the environment and return a Brax `State`.

        The key and parameters are stored in `state.info` for use by `step`.
        """
        if params is None:
            params = self.env.default_params
        obs, env_state = self.env.reset(key, params)
        return State(
            pipeline_state=env_state,
            obs=obs,
            reward=jax.numpy.array(0.0),
            done=jax.numpy.array(False),
            metrics={},
            info={
                "_key": jax.random.split(key)[0],
                "_env_params": params,
                "terminated": jax.numpy.array(False),
                "truncated": jax.numpy.array(False),
                "discount": jax.numpy.array(1.0),
                "final_observation": obs,
            },
        )

    def step(
        self,
        state,  #: State,
        action,  #: Union[jax.Array, jax.Array],
        params=None,  #: Optional[environment.EnvParams] = None,
    ):  # -> State:
        """Advance a Brax `State` by one step.

        Uses the key stored in `state.info`. If `params` is given it replaces the
        stored parameters.
        """
        key, step_key = jax.random.split(state.info["_key"])
        if params is None:
            params = state.info["_env_params"]
        o, env_state, r, terminated, truncated, env_info = self.env.step(
            step_key, state.pipeline_state, action, params
        )
        done = jax.numpy.logical_or(terminated, truncated)
        info = {
            **state.info,
            **env_info,
            "_key": key,
            "_env_params": params,
            "terminated": terminated,
            "truncated": truncated,
        }
        return state.replace(
            pipeline_state=env_state,
            obs=o,
            reward=jax.numpy.array(r),
            done=jax.numpy.array(done),
            info=info,
        )

    def action_size(self) -> int:
        """Number of leaves in an action sampled under the default parameters."""
        a_space = self.env.action_space(self.env.default_params)
        example_a = a_space.sample(jax.random.key(0))
        return len(jax.tree.flatten(example_a)[0])

    def observation_size(self) -> int:
        """Number of leaves in an observation sampled under the default parameters."""
        o_space = self.env.observation_space(self.env.default_params)
        example_o = o_space.sample(jax.random.key(0))
        return len(jax.tree.flatten(example_o)[0])

    def backend(self) -> str:
        """Name of the backend, always `"jax"`."""
        return "jax"

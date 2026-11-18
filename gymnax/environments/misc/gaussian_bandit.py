"""JAX implementation of a two-armed Gaussian bandit for meta-learning."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the GaussianBandit environment.

    Attributes:
        last_action: Arm pulled in the previous step (0 after a reset).
        last_reward: Reward received in the previous step (0 after a reset).
        mu: Mean reward of the second arm in the current episode.
        time: Number of steps taken in the current episode.
    """

    last_action: int
    last_reward: jax.Array
    mu: jax.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the GaussianBandit environment.

    Attributes:
        mean_mu: Mean of the Gaussian from which the second arm's mean reward `mu`
            is sampled at every reset.
        sigma_p: Standard deviation of `mu` across episodes.
        sigma_l: Standard deviation of the second arm's reward across pulls.
        normalize_time: Whether the time step in the observation is rescaled so
            that steps 0 and 100 map to -1 and 1, instead of given as a raw step
            count.
        max_steps_in_episode: Number of steps after which the episode is truncated.
    """

    mean_mu: float = -1.0
    sigma_p: float = 1.0
    sigma_l: float = 0.1
    normalize_time: bool = True
    max_steps_in_episode: int = 100


class GaussianBandit(environment.Environment[EnvState, EnvParams]):
    """Decide whether a stochastic Gaussian arm beats a fixed zero-reward arm.

    Two-armed Gaussian bandit for meta-learning as in Lange & Sprekeler (2022).
    The first arm always pays a reward of 0. The mean `mu` of the second arm is
    sampled at every reset from a Gaussian with mean `params.mean_mu` and standard
    deviation `params.sigma_p` and stays fixed for the whole episode.

    - Observation: `Box` of shape `(4,)` with the last reward, the one-hot encoded
      last action and the time step (normalized if `params.normalize_time`).
    - Actions: `Discrete(2)`. 0 pulls the fixed arm, 1 the stochastic arm.
    - Reward: 0 for the first arm. For the second arm a sample from a Gaussian
      with mean `mu` and standard deviation `params.sigma_l`.
    - Termination: never.
    - Truncation: after `params.max_steps_in_episode` steps (100 by default).

    References:
        - [Lange & Sprekeler (2022)](https://arxiv.org/abs/2010.04466)
    """

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for GaussianBandit-misc."""
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Sample Gaussian reward, increase counter, construct input."""
        # Reparametrization sampling of reward
        reward_arm_1 = 0.0
        reward_arm_2 = (
            state.mu + jax.random.normal(key, ()).astype(jnp.float32) * params.sigma_l
        )
        reward = jax.lax.select(action == 0, reward_arm_1, reward_arm_2)

        # Update state dict and evaluate termination conditions
        state = EnvState(
            last_action=action, last_reward=reward, mu=state.mu, time=state.time + 1
        )
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling the mean of the second arm."""
        # Sample reward function + construct state as concat with timestamp
        mu = (
            params.mean_mu
            + jax.random.normal(key, ()).astype(jnp.float32) * params.sigma_p
        )

        state = EnvState(last_action=0, last_reward=jnp.array(0.0), mu=mu, time=0.0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Concatenate reward, one-hot action and time stamp."""
        action_one_hot = jax.nn.one_hot(state.last_action, self.num_actions).squeeze()
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        return jnp.hstack([state.last_reward, action_one_hot, time_rep])

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "GaussianBandit-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [0, 0, 0, 0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [self.num_actions, 1, 1, params.max_steps_in_episode],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (4,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions),
                "last_reward": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "mu": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def time_normalization(
    t: float, min_lim: float = -1.0, max_lim: float = 1.0, t_max: int = 100
) -> float:
    """Linearly rescale `t` so that 0 maps to `min_lim` and `t_max` to `max_lim`."""
    return (max_lim - min_lim) * t / t_max + min_lim

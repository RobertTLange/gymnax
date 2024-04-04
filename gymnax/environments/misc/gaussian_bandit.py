"""Gaussian bandit environment as in Lange & Sprekeler (2022)."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    last_action: int
    last_reward: jnp.ndarray
    mu: jnp.ndarray
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    mean_mu: float = -1.0  # Mean of stochastic arm
    sigma_p: float = 1.0  # Standard deviation between 'episode'
    sigma_l: float = 0.1  # Standard deviation between 'pulls'
    normalize_time: bool = True
    max_steps_in_episode: int = 100


class GaussianBandit(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of Gaussian bandit environment as in Lange & Sprekeler (2022).


    - 2 arm bandit in which the first arm is fixed to have 0 mean and variance
    - Second arms mean (mu) is sampled from Gaussian with mean -1 and standard
    deviation sigma_p
    - The trial reward for the second arm is then again sampled from a Gaussian w.
    mean
      mu and std sigma_l.
    """

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Sample bernoulli reward, increase counter, construct input."""
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
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Sample reward function + construct state as concat with timestamp
        mu = (
            params.mean_mu
            + jax.random.normal(key, ()).astype(jnp.float32) * params.sigma_p
        )

        state = EnvState(last_action=0, last_reward=jnp.array(0.0), mu=mu, time=0.0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Concatenate reward, one-hot action and time stamp."""
        action_one_hot = jax.nn.one_hot(state.last_action, self.num_actions).squeeze()
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        return jnp.hstack([state.last_reward, action_one_hot, time_rep])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "GaussianBandit-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
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
    """Normalize time integer into range given max time."""
    return (max_lim - min_lim) * t / t_max + min_lim

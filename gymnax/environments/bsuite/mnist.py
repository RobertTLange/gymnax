"""MNIST bandit environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from gymnax.utils import load_mnist


@struct.dataclass
class EnvState(environment.EnvState):
    correct_label: chex.Array
    regret: chex.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    optimal_return: int = 1
    max_steps_in_episode: int = 1


class MNISTBandit(environment.Environment[EnvState, EnvParams]):
    """MNIST bandit environment."""

    def __init__(self, fraction: float = 1.0):
        super().__init__()
        # Load the image MNIST data at environment init
        (images, labels), _ = load_mnist.load_mnist()
        self.num_data = int(fraction * len(labels))
        self.image_shape = images.shape[1:]
        self.images = jnp.array(images[: self.num_data])
        self.labels = jnp.array(labels[: self.num_data])

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
        """Perform single timestep state transition."""
        correct = action == state.correct_label
        reward = lax.select(correct, 1.0, -1.0)
        observation = jnp.zeros(shape=self.image_shape, dtype=jnp.float32)
        state = EnvState(
            correct_label=state.correct_label,
            regret=(state.regret + params.optimal_return - reward),
            time=state.time + 1,
        )
        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        idx = jax.random.randint(key, minval=0, maxval=self.num_data, shape=())
        image = self.images[idx].astype(jnp.float32) / 255
        state = EnvState(
            correct_label=self.labels[idx],
            regret=jnp.array(0.0),
            time=0,
        )
        return image, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Every step transition is terminal! No long term credit assignment!
        return jnp.array(True)

    # def get_obs(self, state: EnvState, params=None) -> None:
    #   """Return observation from raw state trafo."""
    #   # Leave empty - not used here!

    @property
    def name(self) -> str:
        """Environment name."""
        return "MNSITBandit-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 10

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(10)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, shape=self.image_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "correct_label": spaces.Discrete(10),
                "regret": spaces.Box(0, 2, shape=()),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

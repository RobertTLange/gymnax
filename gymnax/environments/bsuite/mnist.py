"""JAX implementation of MNIST bandit environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces
from gymnax.utils import load_mnist


@struct.dataclass
class EnvState(environment.EnvState):
    correct_label: jax.Array
    regret: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    optimal_return: int = 1
    max_steps_in_episode: int = 1


class MNISTBandit(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of MNIST bandit environment."""

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
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        correct = action == state.correct_label
        reward = jax.lax.select(correct, 1.0, -1.0)
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
            jax.lax.stop_gradient(observation),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        idx = jax.random.randint(key, minval=0, maxval=self.num_data, shape=())
        image = self.images[idx].astype(jnp.float32) / 255
        state = EnvState(
            correct_label=self.labels[idx],
            regret=jnp.array(0.0),
            time=0,
        )
        return image, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Every step transition is terminal! No long term credit assignment!
        return jnp.array(True)

    @property
    def name(self) -> str:
        """Environment name."""
        return "MNSITBandit-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 10

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
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

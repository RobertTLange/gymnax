import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from gymnax.utils.load_mnist import load_mnist

from typing import Tuple
import chex

Array = chex.Array
PRNGKey = chex.PRNGKey


class MNISTBandit(environment.Environment):
    def __init__(self, fraction: float = 1.0):
        super().__init__()
        # Load the image MNIST data at environment init
        (images, labels), _ = load_mnist()
        self.num_data = int(fraction * len(labels))
        self.image_shape = images.shape[1:]
        self.images = jnp.array(images[: self.num_data])
        self.labels = jnp.array(labels[: self.num_data])

    @property
    def default_params(self):
        # Default environment parameters
        return {
            "optimal_return": 1,
            "max_steps_in_episode": 1,
        }

    def step_env(
        self, key: PRNGKey, state: dict, action: int, params: dict
    ) -> Tuple[Array, dict, float, bool, dict]:
        """Perform single timestep state transition."""
        correct = action == state["correct_label"]
        reward = lax.select(correct, 1.0, -1.0)
        observation = jnp.zeros(shape=self.image_shape, dtype=jnp.float32)
        state = {
            "correct_label": state["correct_label"],
            "regret": (state["regret"] + params["optimal_return"] - reward),
            "time": state["time"] + 1,
        }
        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        state["terminal"] = done
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(self, key: PRNGKey, params: dict) -> Tuple[Array, dict]:
        """Reset environment state by sampling initial position."""
        idx = jax.random.randint(key, minval=0, maxval=self.num_data, shape=())
        image = self.images[idx].astype(jnp.float32) / 255
        state = {
            "correct_label": self.labels[idx],
            "regret": 0.0,
            "time": 0,
            "terminal": False,
        }
        return image, state

    def is_terminal(self, state: dict, params: dict) -> bool:
        """Check whether state is terminal."""
        # done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        # Every step transition is terminal! No long term credit assignment!
        return True

    def get_obs(self, state: dict) -> Array:
        """Return observation from raw state trafo."""
        # Leave empty - not used here!

    @property
    def name(self) -> str:
        """Environment name."""
        return "MNSITBandit-bsuite"

    @property
    def action_space(self):
        """Action space of the environment."""
        return spaces.Discrete(10)

    def observation_space(self, params: dict):
        """Observation space of the environment."""
        return spaces.Box(0, 1, shape=self.image_shape)

    def state_space(self, params: dict):
        """State space of the environment."""
        return spaces.Dict(
            {
                "correct_label": spaces.Discrete(10),
                "regret": spaces.Box(0, 2, shape=()),
                "time": spaces.Discrete(params["max_steps_in_episode"]),
                "terminal": spaces.Discrete(2),
            }
        )

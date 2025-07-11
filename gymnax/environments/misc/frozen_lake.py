from typing import Dict, Tuple, Any
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


START = 0
FROZEN = 1
HOLE = 2
GOAL = 3


@struct.dataclass
class EnvState(environment.EnvState):
    state: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100


class FrozenLake(environment.Environment[EnvState, EnvParams]):
    def __init__(self, map_name="4x4"):
        super().__init__()
        self.obs_shape = (16,)

        self.maps = {
            "4x4": jnp.array(
                [
                    [START, FROZEN, FROZEN, FROZEN],
                    [FROZEN, HOLE, FROZEN, HOLE],
                    [FROZEN, FROZEN, FROZEN, HOLE],
                    [HOLE, FROZEN, FROZEN, GOAL],
                ]
            ),
            "8x8": jnp.array(
                [
                    [START, FROZEN, FROZEN, FROZEN, FROZEN, FROZEN, FROZEN, FROZEN],
                    [FROZEN, FROZEN, FROZEN, FROZEN, FROZEN, FROZEN, FROZEN, FROZEN],
                    [FROZEN, FROZEN, FROZEN, HOLE, FROZEN, FROZEN, FROZEN, FROZEN],
                    [FROZEN, FROZEN, FROZEN, FROZEN, FROZEN, HOLE, FROZEN, FROZEN],
                    [FROZEN, FROZEN, FROZEN, HOLE, FROZEN, FROZEN, FROZEN, FROZEN],
                    [FROZEN, HOLE, HOLE, FROZEN, FROZEN, FROZEN, HOLE, FROZEN],
                    [FROZEN, HOLE, FROZEN, FROZEN, HOLE, FROZEN, HOLE, FROZEN],
                    [FROZEN, FROZEN, FROZEN, HOLE, FROZEN, FROZEN, FROZEN, GOAL],
                ]
            ),
        }

        self.desc = self.maps[map_name]
        self.nrow, self.ncol = self.desc.shape
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self.directions = jnp.array(
            [
                [0, -1],
                [0, 1],
                [1, 0],
                [-1, 0],
            ]
        )

    @property
    def name(self) -> str:
        return "FrozenLake-misc"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:

        time = state.time
        current_state = state.state
        row = current_state // self.ncol
        col = current_state % self.ncol

        def get_next_state(row, col, action):
            new_row = row + self.directions[action][0]
            new_col = col + self.directions[action][1]
            new_row = jnp.clip(new_row, 0, self.nrow - 1)
            new_col = jnp.clip(new_col, 0, self.ncol - 1)
            return new_row, new_col

        key_random, key_action = jax.random.split(key)
        random_action = jax.random.randint(key_action, (), 0, self.n_actions)
        slip = jax.random.uniform(key_random) < 1 / 3
        action = jax.lax.select(slip, random_action, action)

        new_row, new_col = get_next_state(row, col, action)
        new_state = new_row * self.ncol + new_col

        current_cell = self.desc[new_row, new_col]
        done = (current_cell == GOAL) | (current_cell == HOLE)
        reward = (current_cell == GOAL).astype(jnp.float32)

        new_env_state = EnvState(state=new_state, time=time + 1)

        return (
            jax.lax.stop_gradient(self.get_obs(new_env_state)),
            jax.lax.stop_gradient(new_env_state),
            reward,
            done,
            {"discount": self.discount(new_env_state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:

        initial_state = EnvState(state=jnp.array(0), time=0)

        return self.get_obs(initial_state), initial_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        current_state = state.state
        row = current_state // self.ncol
        col = current_state % self.ncol
        current_cell = self.desc[row, col]

        done_goal = current_cell == GOAL
        done_hole = current_cell == HOLE
        done_time = state.time >= params.max_steps_in_episode

        return jnp.logical_or(jnp.logical_or(done_goal, done_hole), done_time)

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        return state.state

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(low=0, high=15, shape=(1,), dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "state": spaces.Discrete(self.n_states),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

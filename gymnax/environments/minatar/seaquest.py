# """JAX compatible version of Seaquest MinAtar environment."""


# from typing import Any, Dict, Optional, Tuple, Union


# import chex
# from flax import struct
# from gymnax.environments import environment
# from gymnax.environments import spaces
# import jax
# from jax import lax
# import jax.numpy as jnp


# @struct.dataclass
# class EnvState(environment.EnvState):
#   """State of the environment."""


#   oxygen: int
#   sub_x: int
#   sub_y: int
#   sub_or: int
#   f_bullet_count: int
#   f_bullets: chex.Array
#   e_bullet_count: int
#   e_bullets: chex.Array
#   e_fish_count: int
#   e_fish: chex.Array
#   e_subs_count: int
#   e_subs: chex.Array
#   diver_count: int
#   divers: chex.Array
#   e_spawn_speed: int
#   e_spawn_timer: int
#   d_spawn_timer: int
#   move_speed: int
#   ramp_index: int
#   shot_timer: int
#   surface: int
#   time: int
#   terminal: bool


# @struct.dataclass
# class EnvParams(environment.EnvParams):
#   ramping: bool = True
#   ramp_interval: int = 100
#   init_spawn_speed: int = 20
#   init_move_interval: int = 5
#   max_oxygen: int = 200
#   diver_spawn_speed: int = 30
#   shot_cool_down: int = 5
#   enemy_shot_interval: int = 10
#   enemy_move_interval: int = 5
#   diver_move_interval: int = 5
#   max_steps_in_episode: int = 1000


# class MinSeaquest(environment.Environment[EnvState, EnvParams]):
#   """JAX Compatible version of Seaquest MinAtar environment.


#   Source:
#   github.com/kenjyoung/MinAtar/blob/master/minatar/environments/seaquest.py


#   ENVIRONMENT DESCRIPTION - 'Seaquest-MinAtar'
#   - Player controls submarine consisting of two cells - front and back.
#   - Player can fire bullets from front of submarine.
#   - Enemies consist of submarines [shoot] and fish [don't shoot].
#   - A reward of +1 is given whenever enemy is struck by bullet and removed.
#   - Player can pick up drivers which increments a bar indicated by a channel.
#   - Player has limited oxygen supply indicated by bar in separate channel.
#   - Oxygen degrades over time. Can be restored:
#   - If player moves to top of screen and has
#     at least 1 rescued driver on board.
#   - When surfacing with less than 6, one diver is removed.
#   - When surfacing w. 6, remove all divers. R for each active cell in oxygen
#   bar.
#   - Each time the player surfaces increase difficulty by increasing
#     the spawn rate and movement speed of enemies.
#   - Termination occurs when player is hit by an enemy fish, sub or bullet
#   - Or when oxygen reached 0.
#   - Or when the layer attempts to surface with no rescued divers.
#   - Enemy and diver directions are indicated by a trail channel active
#   - in their previous location to reduce partial observability.


#   - Channels are encoded as follows: 'sub_front':0, 'sub_back':1,
#                                      'friendly_bullet':2, 'trail':3,
#                                      'enemy_bullet':4, 'enemy_fish':5,
#                                      'enemy_sub':6, 'oxygen_guage':7,
#                                      'diver_guage':8, 'diver':9
#   - Observation has dimensionality (10, 10, 10)
#   - Actions are encoded as follows: ['n','l','u','r','d','f']
#   """


#   def __init__(self, use_minimal_action_set: bool = True):
#     super().__init__()
#     self.obs_shape = (10, 10, 10)
#     # Full action set: ['n','l','u','r','d','f']
#     self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
#     # Minimal action set: ['n','l','u','r','d','f']
#     self.minimal_action_set = jnp.array([0, 1, 2, 3, 4, 5])
#     # Set active action set for environment
#     # If minimal map to integer in full action set
#     if use_minimal_action_set:
#       self.action_set = self.minimal_action_set
#     else:
#       self.action_set = self.full_action_set


#   @property
#   def default_params(self) -> EnvParams:
#     # Default environment parameters
#     return EnvParams()


#   def step_env(
#       self,
#       key: chex.PRNGKey,
#       state: EnvState,
#       action: Union[int, float, chex.Array],
#       params: EnvParams,
#   ) -> Tuple[chex.Array, EnvState, jnp.ndarray, bool, Dict]:  # dict]:
#     """Perform single timestep state transition."""
#     # If timer is up spawn enemy and divers [always sample]
#     # key_enemy, key_diver = jax.random.split(key)
#     # spawned_enemy = spawn_enemy(key_enemy, state, params)
#     spawn_enemy_cond = state.e_spawn_timer == 0
#     state = state.replace(
#         e_spawn_timer=lax.select(
#             spawn_enemy_cond, state.e_spawn_speed, state.e_spawn_timer
#         )
#     )


#     # spawned_diver = spawn_diver(key_diver, params)
#     spawn_diver_cond = state.d_spawn_timer == 0
#     state = state.replace(
#         d_spawn_timer=lax.select(
#             spawn_diver_cond, state.d_spawn_speed, state.d_spawn_timer
#         )
#     )


#     # Sequentially go through substate and update the state
#     a = self.action_set[action]
#     state = step_agent(state, a, self.env_params)
#     reward = step_bullets(state)
#     state = step_divers(state)
#     state, reward = step_e_subs(state, reward)
#     state, reward = step_e_bullets(state, reward)
#     state, reward = step_timers(state, reward, params)
#     # Check game condition & no. steps for termination condition
#     state.replace(time=state.time + 1)
#     done = self.is_terminal(state, params)
#     state = state.replace(terminal=done)
#     info = {"discount": self.discount(state, params)}
#     return (
#         lax.stop_gradient(self.get_obs(state, params)),
#         lax.stop_gradient(state),
#         reward.astype(jnp.float32),
#         done,
#         info,
#     )


#   def reset_env(
#       self, key: chex.PRNGKey, params: EnvParams
#   ) -> Tuple[chex.Array, EnvState]:
#     """Reset environment state by sampling initial position."""
#     state = EnvState(
#         oxygen=params.max_oxygen,
#         sub_x=5,
#         sub_y=0,
#         sub_or=0,
#         f_bullet_count=0,
#         f_bullets=jnp.zeros((100, 3), dtype=jnp.int32),
#         e_bullet_count=0,
#         e_bullets=jnp.zeros((100, 3), dtype=jnp.int32),
#         e_fish_count=0,
#         e_fish=jnp.zeros((100, 5), dtype=jnp.int32),
#         e_subs_count=0,
#         e_subs=jnp.zeros((100, 5), dtype=jnp.int32),
#         diver_count=0,
#         divers=jnp.zeros((100, 4), dtype=jnp.int32),
#         e_spawn_speed=params.init_spawn_speed,
#         e_spawn_timer=params.init_spawn_speed,
#         d_spawn_timer=params.diver_spawn_speed,
#         move_speed=params.init_move_interval,
#         ramp_index=0,
#         shot_timer=0,
#         surface=1,
#         time=0,
#         terminal=False,
#     )
#     return self.get_obs(state, params), state


#   def get_obs(self, state: EnvState, params: EnvParams,
#     key=None) -> chex.Array:
#     """Return observation from raw state trafo."""
#     fish, sub, diver = [], [], []
#     obs = jnp.zeros(self.obs_shape, dtype=bool)
#     # Set agents sub-front and back, oxygen_gauge and diver_gauge
#     obs = obs.at[state.sub_y, state.sub_x, 0].set(1)
#     back_x = (state.sub_x - 1) * state.sub_or + (state.sub_x + 1) * (
#         1 - state.sub_or
#     )
#     obs = obs.at[state.sub_y, back_x, 1].set(1)
#     obs = obs.at[9, 0 : state.oxygen * 10 // params.max_oxygen, 7].set(1)
#     obs = obs.at[9, 9 - state.diver_count : 9, 8].set(1)


#     # Set friendly bulltes, enemy bullets, enemy fish+trail, enemey sub+trail
#     for f_b_id in range(state.f_bullet_count):
#       obs = obs.at[
#           state.f_bullets[f_b_id, 1],
#           state.f_bullets[f_b_id, 0],
#           2,
#       ].set(1)
#     for e_b_id in range(state.e_bullet_count):
#       obs = obs.at[
#           state.e_bullets[e_b_id, 1],
#           state.e_bullets[e_b_id, 0],
#           4,
#       ].set(1)
#     for e_f_id in range(state.e_fish_count):
#       obs = obs.at[state.e_fish[e_f_id, 1], state.e_fish[e_f_id, 0], 5].set(1)
#       back_x = (fish[0] - 1) * fish[2] + (fish[0] + 1) * (1 - fish[2])
#       border_cond = jnp.logical_and(back_x >= 0, back_x <= 9)
#       obs = jax.lax.select(
#           border_cond,
#           obs.at[state.e_fish[e_f_id][1], back_x, 3].set(1),
#           obs,
#       )


#     for e_s_id in range(state.e_subs_count):
#       obs = obs.at[state.e_subs[e_s_id, 1], state.e_subs[e_s_id, 0], 6].set(1)
#       back_x = (sub[0] - 1) * sub[2] + (sub[0] + 1) * (1 - sub[2])
#       border_cond = jnp.logical_and(back_x >= 0, back_x <= 9)
#       obs = jax.lax.select(
#           border_cond,
#           obs.at[state.e_subs[e_s_id, 1], back_x, 3].set(1),
#           obs,
#       )


#     for d_id in range(state.diver_count):
#       obs = obs.at[state.divers[d_id, 1], state.divers[d_id, 0], 9].set(1)
#       back_x = (diver[0] - 1) * diver[2] + (diver[0] + 1) * (1 - diver[2])
#       border_cond = jnp.logical_and(back_x >= 0, back_x <= 9)
#       obs = jax.lax.select(
#           border_cond,
#           obs.at[state.divers[d_id, 1], back_x, 3].set(1),
#           obs,
#       )
#     return obs.astype(jnp.float32)


#   def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
#     """Check whether state is terminal."""
#     done_steps = state.time >= params.max_steps_in_episode
#     return jnp.logical_or(state.terminal, done_steps).item()


#   @property
#   def name(self) -> str:
#     """Environment name."""
#     return "Seaquest-MinAtar"


#   @property
#   def num_actions(self) -> int:
#     """Number of actions possible in environment."""
#     return len(self.action_set)


#   def action_space(self,
#     params: Optional[EnvParams] = None) -> spaces.Discrete:
#     """Action space of the environment."""
#     return spaces.Discrete(len(self.action_set))


#   def observation_space(self, params: EnvParams) -> spaces.Box:
#     """Observation space of the environment."""
#     return spaces.Box(0, 1, self.obs_shape)


#   def state_space(self, params: EnvParams) -> spaces.Dict:
#     """State space of the environment."""
#     return spaces.Dict({
#         "oxygen": spaces.Discrete(params.max_oxygen),
#         "diver_count": spaces.Discrete(20),
#         "sub_x": spaces.Discrete(10),
#         "sub_y": spaces.Discrete(10),
#         "sub_or": spaces.Discrete(2),
#         "f_bullets": spaces.Box(0, 1, (100, 3)),
#         "e_bullets": spaces.Box(0, 1, (100, 3)),
#         "e_fish": spaces.Box(0, 1, (100, 5)),
#         "e_subs": spaces.Box(0, 1, (100, 5)),
#         "divers": spaces.Box(0, 1, (100, 4)),
#         "e_spawn_speed": spaces.Discrete(params.init_spawn_speed),
#         "e_spawn_timer": spaces.Discrete(params.init_spawn_speed),
#         "d_spawn_timer": spaces.Discrete(params.diver_spawn_speed),
#         "move_speed": spaces.Discrete(1000),
#         "ramp_index": spaces.Discrete(1000),
#         "shot_timer": spaces.Discrete(params.shot_cool_down),
#         "surface": spaces.Discrete(2),
#         "time": spaces.Discrete(params.max_steps_in_episode),
#         "terminal": spaces.Discrete(2),
#     })


# def step_agent(state: EnvState, action: int,
#   env_params: EnvParams) -> EnvState:
#   """Perform submarine position and friendly bullets transition."""
#   # Update submarine position based on l, r or u, d actions
#   not_l_or_r = jnp.logical_and(action != 1, action != 3)
#   state["sub_x"] = (
#       (action == 1) * jnp.maximum(0, state["sub_x"] - 1)
#       + (action == 3) * jnp.minimum(9, state["sub_x"] + 1)
#       + not_l_or_r * state["sub_x"]
#   )
#   state["sub_or"] = lax.select(action == 1, False, state["sub_or"])
#   state["sub_or"] = lax.select(action == 3, True, state["sub_or"])


#   not_u_or_d = jnp.logical_and(action != 2, action != 4)
#   state["sub_y"] = (
#       (action == 2) * jnp.maximum(0, state["sub_y"] - 1)
#       + (action == 4) * jnp.minimum(8, state["sub_y"] + 1)
#       + not_u_or_d * state["sub_y"]
#   )


#   # Update friendly bullets based on f action and shot_timer
#   bullet_cond = jnp.logical_and(action == 5, state["shot_timer"] == 0)
#   state["shot_timer"] = lax.select(
#       bullet_cond, env_params.shot_cool_down, state["shot_timer"]
#   )
#   bullet_array = jnp.array([state["sub_x"], state["sub_y"], state["sub_or"]])
#   # Use counter to keep track of row idx to update!
#   f_bullets_add = jax.ops.index_update(
#       state["f_bullets"], jax.ops.index[state["f_bullet_count"]], bullet_array
#   )
#   state["f_bullets"] = lax.select(
#       bullet_cond, f_bullets_add, state["f_bullets"]
#   )
#   state["f_bullet_count"] += bullet_cond
#   return state


# def step_bullets(state: EnvState) -> Tuple[EnvState, float]:
#   """Perform friendly bullets transition."""
#   reward = 0.0
#   f_bullet_count = 0
#   f_bullets = jnp.zeros((100, 3))
#   #   e_fish = jnp.zeros((100, 3))
#   for f_b_id in range(state["f_bullet_count"]):
#     bullet_to_check = state["f_bullets"][f_b_id].copy()
#     bullet_to_check[0] = lax.select(
#         bullet_to_check[2], bullet_to_check[0] + 1, bullet_to_check[0] - 1
#     )


#     # Add bullet if it has not exited
#     bullet_border = jnp.logical_or(
#         bullet_to_check[0] < 0, bullet_to_check[0] > 9
#     )
#     f_bullets = jax.ops.index_update(
#         f_bullets,
#         jax.ops.index(f_bullet_count),
#         bullet_to_check * (1 - bullet_border),
#     )
#     f_bullet_count += 1 - bullet_border


#     # Check for collision with enemy fish
#     # removed = 0
#     # for e_f_id in range(state["e_fish_count"]):
#     #   e_fish_to_check = state["e_fish"][e_f_id].copy()
#     #   hit = state["f_bullets"][f_b_id][0:2] == state["e_fish"][e_f_id][0:2]


#   return state, reward


# def collision_and_remove(indiv_to_check, entity_counter, entities):
#   """Helper function that checks for collision and updates entities."""
#   entities_clean = jnp.zeros(entities.shape)
#   entity_counter_clean = 0
#   for e_id in range(entity_counter):
#     hit = (indiv_to_check[0:2] == entities[e_id][0:2]).all()
#     # If no hit - add entity to array and increase clean counter
#     entities_clean = jax.ops.index_update(
#         entities_clean,
#         jax.ops.index(entity_counter_clean),
#         entities[e_id] * (1 - hit),
#     )
#     entity_counter_clean += hit
#   return


# def step_divers(state):
#   """Perform diver transition."""
#   return state


# def step_e_subs(state, reward):
#   """Perform enemy submarine transition."""
#   return state, reward


# def step_e_bullets(state, reward):
#   """Perform enemy bullets and enemy fish transition."""
#   return state, reward


# def spawn_enemy(
#     key: chex.PRNGKey, state: EnvState, env_params: EnvParams
# ) -> chex.Array:
#   """Spawn a new enemy."""
#   lr_key, sub_key, y_key = jax.random.splt(key, 3)
#   lr = jax.random.choice(lr_key, 2, ())
#   is_sub = jax.random.choice(sub_key, 2, (), p=jnp.array([1 / 3, 2 / 3]))
#   x = lax.select(lr, 0, 9)
#   y = jax.random.choice(y_key, jnp.arange(1, 9), ())


#   # # Do not spawn in same row an opposite direction as existing
#   # if(any([z[1]==y and z[2]!=lr for z in self.e_subs+self.e_fish])):
#   #     return
#   # if(is_sub):
#   #     self.e_subs+=[[x,y,lr,self.move_speed,enemy_shot_interval]]
#   # else:
#   #     self.e_fish+=[[x,y,lr,self.move_speed]]
#   return jnp.array([
#       is_sub,
#       x,
#       y,
#       lr,
#       state["move_speed"],
#       env_params["enemy_shot_interval"],
#   ])


# def spawn_diver(key: chex.PRNGKey, env_params: EnvParams) -> chex.Array:
#   """Spawn a new diver."""
#   lr_key, y_key = jax.random.splt(key)
#   lr = jax.random.choice(lr_key, 2, ())
#   x = lax.select(lr, 0, 9)
#   y = jax.random.choice(y_key, jnp.arange(1, 9), ())
#   return jnp.array([x, y, lr, env_params.diver_move_interval])


# def step_timers(state: EnvState, reward: float, env_params: EnvParams):
#   """Update the timers of the environment and calculate surface reward."""
#   #   e_spawn_timer = state.e_spawn_timer - state.e_spawn_timer > 0
#   #   d_spawn_timer = state.d_spawn_timer - state.d_spawn_timer > 0
#   #   shot_timer = state.shot_timer - state.shot_timer > 0
#   #   oxy_term = lax.select(state.oxygen < 0, 1, 0)


#   # Update oxygen and surface indicator if submarine is above
#   above_surface = state.sub_y > 0
#   #   oxygen = lax.select(above_surface, state.oxygen - 1, state.oxygen)
#   surface_val = lax.select(above_surface, 1, 0)


#   # Calculate reward/terminate episode otherwise
#   below_cond = jnp.logical_and(1 - above_surface, 1 - surface_val)
#   #   diver_term = jnp.logical_and(below_cond, state.diver_count == 0)
#   surface_cond = jnp.logical_and(below_cond, 1 - (state.diver_count == 0))
#   state, surface_reward = surface(surface_cond, state, env_params)
#   reward += surface_cond * surface_reward
#   return state, reward


# def surface(
#     surface_cond: bool, state: EnvState, env_params: EnvParams
# ) -> Tuple[EnvState, float]:
#   """Perform surface transition and reward calculations."""
#   # surface = 1
#   diver_count = lax.select(state.diver_count == 6, 0, state.diver_count)
#   reward = lax.select(
#       state.diver_count == 6,
#       state.oxygen * 10 // env_params.max_oxygen,
#       0,
#   )
#   oxygen = state.oxygen
#   diver_count -= 1
#   ramp_cond = jnp.logical_and(
#       env_params.ramping,
#       jnp.logical_or(state.e_spawn_speed > 1, state.move_speed > 2),
#   )
#   move_cond = jnp.logical_and(
#       ramp_cond,
#       jnp.logical_and(state.move_speed > 2, state.ramp_index % 2),
#   )
#   move_speed = lax.select(move_cond, state.move_speed - 1, state.move_speed)
#   e_spawn_cond = jnp.logical_and(ramp_cond, state.e_spawn_speed > 1)
#   e_spawn_speed = lax.select(
#       e_spawn_cond, state.e_spawn_speed - 1, state.e_spawn_speed
#   )


#   # Update the state based on the surface_cond - only update if cond met!
#   state.diver_count = lax.select(surface_cond, diver_count, state.diver_count)
#   state.oxygen = lax.select(surface_cond, oxygen, state.oxygen)
#   state.move_speed = lax.select(surface_cond, move_speed, state.move_speed)
#   state.e_spawn_speed = lax.select(
#       surface_cond, e_spawn_speed, state.e_spawn_speed
#   )
#   return state, reward

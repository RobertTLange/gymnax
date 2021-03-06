import jax
import jax.numpy as jnp
from jax import jit

# JAX Compatible version of Asterix MinAtar environment. Source:
# github.com/kenjyoung/MinAtar/blob/master/minatar/environments/asterix.py

# Default environment parameters of Asterix game
params_asterix = {"ramping": 1,
                  "ramp_interval": 100,
                  "init_spawn_speed": 10,
                  "init_move_interval": 5,
                  "shot_cool_down": 5}

"""
- Player moves freely along 4 cardinal dirs.
- Enemies and treasure spawn from the sides.
- A reward of +1 is given for picking up treasure.
- Termination occurs if the player makes contact with an enemy.
- Enemy and treasure direction are indicated by a trail channel.
- Difficulty periodically increases: the speed/spawn rate of enemies/treasure.
- Channels are encoded as follows: 'player':0, 'enemy':1, 'trail':2, 'gold':3
- Observation has dimensionality (10, 10, 4)
- Actions are encoded as follows: 'l': 0, 'u': 1, 'r': 2, 'd': 3.
- Note that is different from MinAtar where 0 is 'n' and 5 is 'f' (None/Fire).
"""


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    reward = 0

    # Spawn enemy if timer is up
    entity, slot = spawn_entity(rng_input, state)
    if(self.spawn_timer==0):
        state["entities"][slot] = entity
        state["spawn_timer"] = state["spawn_speed"]

    # Resolve player action
    if(a=='l'):
        player_x = max(0, player_x-1)
    elif(a=='r'):
        player_x = min(9, player_x+1)
    elif(a=='u'):
        player_y = max(1, player_y-1)
    elif(a=='d'):
        player_y = min(8, player_y+1)

    # Update entities
    for i in range(len(state["entities"])):
        x = state["entities"]
        if(x is not None):
            if(x[0:2]==[self.player_x, self.player_y]):
                if(self.entities[i][3]):
                    self.entities[i] = None
                    r+=1
                else:
                    self.terminal = True
    if(self.move_timer==0):
        self.move_timer = self.move_speed
        for i in range(len(self.entities)):
            x = self.entities[i]
            if(x is not None):
                x[0]+=1 if x[2] else -1
                if(x[0]<0 or x[0]>9):
                    self.entities[i] = None
                if(x[0:2]==[self.player_x,self.player_y]):
                    if(self.entities[i][3]):
                        self.entities[i] = None
                        r+=1
                    else:
                        self.terminal = True

    # Update various timers
    self.spawn_timer -= 1
    self.move_timer -= 1


    #Ramp difficulty if interval has elapsed
    if self.ramping and (self.spawn_speed>1 or self.move_speed>1):
        if(self.ramp_timer>=0):
            self.ramp_timer-=1
        else:
            if(self.move_speed>1 and self.ramp_index%2):
                self.move_speed-=1
            if(self.spawn_speed>1):
                self.spawn_speed-=1
            self.ramp_index+=1
            self.ramp_timer=ramp_interval
    return get_obs(state), state, reward, done, info


def reset(rng_input, params):
    """ Reset environment state by sampling initial position. """
    state = {
        "player_state": jnp.array([5, 5]),
        "shot_timer": 0,
        "spawn_speed": params["init_spawn_speed"],
        "spawn_times": params["init_spawn_speed"],
        "move_speed": params["init_move_interval"],
        "move_timer": params["init_move_interval"],
        "ramp_timer": params["ramp_interval"],
        "ramp_index": 0,
        "terminal": 0,
        "entities": 8*[None]
    }
    return get_obs(state), state


def get_obs(state):
    """ Return observation from raw state trafo. """
    obs = jnp.zeros((10, 10, 4), dtype=bool)
    obs = jax.ops.index_update(obs, jax.ops.index[state["player_state"][0],
                                                  state["player_state"][1],
                                                  0], 1)
    not_none_entities = [i for i in range(len(state["entities"]))
                         if state["entities"][i] is not None]
    for i in not_none_entities:
        x = state["entities"][i]
        c = 3 * x[3] + 1 * (1 - x[3])
        obs = jax.ops.index_update(obs, jax.ops.index[x[1], x[0], c], 1)
        back_x = (x[0] - 1) * x[2] + (x[0] + 1) * (1 - x[2])
        leave_trail = jnp.logical_and(back_x >= 0, back_x<=9)
        obs = jax.ops.index_update(obs, jax.ops.index[x[1], back_x, 2],
                                   leave_trail)
    return obs


def spawn_entity(rng_input, state):
    """
        Spawn new enemy or treasure at random location
        with random direction (if all rows are filled do nothing).
    """
    key_lr, key_gold, key_slot = jax.random.split(rng_input, 3)
    lr = jax.random.choice(key_lr, [True, False])
    is_gold = jax.random.choice(key_gold, [True, False], p=[1/3, 2/3])
    x = (1 - lr) * 9
    slot_options = [i for i in range(len(state["entities"]))
                    if state["entities"] is None]
    slot = jax.random.choice(key_slot, slot_options)
    entity = [x, slot+1, lr, is_gold]
    return slot, entity


reset_asterix = jit(reset)
step_asterix = jit(step)

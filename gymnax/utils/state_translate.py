"""Helper functions to translate gym state to JAX state."""

import jax.numpy as jnp
from gymnax.environments.bsuite import bandit
from gymnax.environments.bsuite import catch
from gymnax.environments.bsuite import deep_sea
from gymnax.environments.bsuite import discounting_chain
from gymnax.environments.bsuite import memory_chain
from gymnax.environments.bsuite import mnist
from gymnax.environments.bsuite import umbrella_chain
from gymnax.environments.classic_control import acrobot
from gymnax.environments.classic_control import cartpole
from gymnax.environments.classic_control import continuous_mountain_car
from gymnax.environments.classic_control import mountain_car
from gymnax.environments.classic_control import pendulum
from gymnax.environments.minatar import asterix
from gymnax.environments.minatar import breakout
from gymnax.environments.minatar import freeway
from gymnax.environments.minatar import space_invaders

# from gymnax.environments.minatar import seaquest


def np_state_to_jax(env, env_name: str = "Pendulum-v1", get_jax: bool = False):
    """Helper that collects env state into dict for JAX `step`."""

    if env_name in [
        "Pendulum-v1",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Acrobot-v1",
    ]:
        state_gym_to_jax = control_np_to_jax(env, env_name, get_jax)
    elif env_name in [
        "Catch-bsuite",
        "DeepSea-bsuite",
        "DiscountingChain-bsuite",
        "MemoryChain-bsuite",
        "UmbrellaChain-bsuite",
        "MNISTBandit-bsuite",
        "SimpleBandit-bsuite",
    ]:
        state_gym_to_jax = bsuite_np_to_jax(env, env_name, get_jax)
    elif env_name in [
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        # "Seaquest-MinAtar",
        "SpaceInvaders-MinAtar",
    ]:
        state_gym_to_jax = minatar_np_to_jax(env, env_name, get_jax)
    else:
        raise ValueError(f"{env_name} is not in set of implemented environments.")
    return state_gym_to_jax


def control_np_to_jax(env, env_name: str = "Pendulum-v1", get_jax: bool = False):
    """Collects env state of classic_control into dict for JAX `step`."""
    state_gym_to_jax = None
    if env_name == "Pendulum-v1":
        state_gym_to_jax = {
            "theta": env.state[0],
            "theta_dot": env.state[1],
            "last_u": env.last_u,
            "time": 0,
        }
        if get_jax:
            return pendulum.EnvState(**state_gym_to_jax)
    elif env_name == "CartPole-v1":
        state_gym_to_jax = {
            "x": env.state[0],
            "x_dot": env.state[1],
            "theta": env.state[2],
            "theta_dot": env.state[3],
            "time": 0,
        }
        if get_jax:
            return cartpole.EnvState(**state_gym_to_jax)
    elif env_name == "MountainCar-v0":
        state_gym_to_jax = {
            "position": env.state[0],
            "velocity": env.state[1],
            "time": 0,
        }
        if get_jax:
            return mountain_car.EnvState(**state_gym_to_jax)
    elif env_name == "MountainCarContinuous-v0":
        state_gym_to_jax = {
            "position": env.state[0],
            "velocity": env.state[1],
            "time": 0,
        }
        if get_jax:

            return continuous_mountain_car.EnvState(**state_gym_to_jax)
    elif env_name == "Acrobot-v1":

        state_gym_to_jax = {
            "joint_angle1": env.state[0],
            "joint_angle2": env.state[1],
            "velocity_1": env.state[2],
            "velocity_2": env.state[3],
            "time": 0,
        }
        if get_jax:

            return acrobot.EnvState(**state_gym_to_jax)
    return state_gym_to_jax


def bsuite_np_to_jax(env, env_name: str = "Catch-bsuite", get_jax: bool = False):
    """Collects env state of bsuite into dict for JAX `step`."""
    state_gym_to_jax = None
    if env_name == "Catch-bsuite":
        state_gym_to_jax = {
            "ball_x": env._ball_x,
            "ball_y": env._ball_y,
            "paddle_x": env._paddle_x,
            "paddle_y": env._paddle_y,
            "prev_done": env._reset_next_step,
            "time": 0,
        }
        if get_jax:

            return catch.EnvState(**state_gym_to_jax)
    elif env_name == "DeepSea-bsuite":
        state_gym_to_jax = {
            "row": env._row,
            "column": env._column,
            "bad_episode": env._bad_episode,
            "total_bad_episodes": env._total_bad_episodes,
            "denoised_return": env._denoised_return,
            "optimal_return": env._optimal_return,
            "action_mapping": env._action_mapping,
            "time": 0,
        }
        if get_jax:

            return deep_sea.EnvState(**state_gym_to_jax)
    elif env_name == "DiscountingChain-bsuite":
        state_gym_to_jax = {
            "rewards": env._rewards,
            "context": env._context,
            "time": env._timestep,
        }
        if get_jax:

            return discounting_chain.EnvState(**state_gym_to_jax)
    elif env_name == "MemoryChain-bsuite":
        state_gym_to_jax = {
            "context": env._context,
            "query": env._query,
            "total_perfect": env._total_perfect,
            "total_regret": env._total_regret,
            "time": env._timestep,
        }
        if get_jax:

            return memory_chain.EnvState(**state_gym_to_jax)
    elif env_name == "UmbrellaChain-bsuite":
        state_gym_to_jax = {
            "need_umbrella": env._need_umbrella,
            "has_umbrella": env._has_umbrella,
            "total_regret": env._total_regret,
            "time": env._timestep,
        }
        if get_jax:

            return umbrella_chain.EnvState(**state_gym_to_jax)
    elif env_name == "MNISTBandit-bsuite":
        state_gym_to_jax = {
            "correct_label": env._correct_label,
            "regret": env._total_regret,
            "time": 0,
        }
        if get_jax:

            return mnist.EnvState(**state_gym_to_jax)
    elif env_name == "SimpleBandit-bsuite":
        state_gym_to_jax = {
            "rewards": env._rewards,
            "total_regret": env._total_regret,
            "time": 0,
        }
        if get_jax:

            return bandit.EnvState(**state_gym_to_jax)
    return state_gym_to_jax


def minatar_np_to_jax(env, env_name: str = "Asterix-MinAtar", get_jax: bool = False):
    """Collects env state of MinAtar into dict for JAX `step`."""
    state_gym_to_jax = None
    if env_name == "Asterix-MinAtar":
        entities_array = jnp.zeros((8, 5), dtype=jnp.int32)
        for i in range(8):
            if env.env.entities[i] is not None:
                entities_array = entities_array.at[i, 0:4].set(env.env.entities[i])
                entities_array = entities_array.at[i, 4].set(1)
        state_gym_to_jax = {
            "player_x": env.env.player_x,
            "player_y": env.env.player_y,
            "shot_timer": env.env.shot_timer,
            "spawn_speed": env.env.spawn_speed,
            "spawn_timer": env.env.spawn_timer,
            "move_speed": env.env.move_speed,
            "move_timer": env.env.move_timer,
            "ramp_timer": env.env.ramp_timer,
            "ramp_index": env.env.ramp_index,
            "entities": entities_array,
            "time": 0,
            "terminal": False,
        }
        if get_jax:

            return asterix.EnvState(**state_gym_to_jax)
    elif env_name == "Breakout-MinAtar":
        state_gym_to_jax = {
            "ball_y": jnp.array(env.env.ball_y),
            "ball_x": jnp.array(env.env.ball_x),
            "ball_dir": env.env.ball_dir,
            "pos": env.env.pos,
            "brick_map": jnp.array(env.env.brick_map),
            "strike": env.env.strike,
            "last_y": jnp.array(env.env.last_y),
            "last_x": jnp.array(env.env.last_x),
            "time": 0,
            "terminal": False,
        }
        if get_jax:

            return breakout.EnvState(**state_gym_to_jax)
    elif env_name == "Freeway-MinAtar":
        state_gym_to_jax = {
            "pos": env.env.pos,
            "cars": jnp.array(env.env.cars),
            "move_timer": env.env.move_timer,
            "time": 0,
            "terminal": False,
        }
        if get_jax:

            return freeway.EnvState(**state_gym_to_jax)
    # elif env_name == "Seaquest-MinAtar":
    #   f_bullets = np.zeros((100, 3))
    #   for i, f_b in enumerate(env.env.f_bullets):
    #     f_bullets[i] = f_b
    #   e_bullets = np.zeros((100, 3))
    #   for i, e_b in enumerate(env.env.e_bullets):
    #     e_bullets[i] = e_b
    #   e_fish = np.zeros((100, 5))
    #   for i, e_f in enumerate(env.env.e_fish):
    #     e_fish[i] = e_f + [10]
    #   e_subs = np.zeros((100, 5))
    #   for i, e_s in enumerate(env.env.e_subs):
    #     e_subs[i] = e_s
    #   divers = np.zeros((100, 4))
    #   for i, d in enumerate(env.env.divers):
    #     divers[i] = d

    #   state_gym_to_jax = {
    #       "oxygen": env.env.oxygen,
    #       "sub_x": env.env.sub_x,
    #       "sub_y": env.env.sub_y,
    #       "sub_or": env.env.sub_or,
    #       "f_bullet_count": len(env.env.f_bullets),
    #       "f_bullets": f_bullets,
    #       "e_bullet_count": len(env.env.e_bullets),
    #       "e_bullets": e_bullets,
    #       "e_fish_count": len(env.env.e_fish),
    #       "e_fish": e_fish,
    #       "e_subs_count": len(env.env.e_subs),
    #       "e_subs": e_subs,
    #       "diver_count": env.env.diver_count,
    #       "divers": divers,
    #       "e_spawn_speed": env.env.e_spawn_speed,
    #       "e_spawn_timer": env.env.e_spawn_timer,
    #       "d_spawn_timer": env.env.d_spawn_timer,
    #       "move_speed": env.env.move_speed,
    #       "ramp_index": env.env.ramp_index,
    #       "shot_timer": env.env.shot_timer,
    #       "surface": env.env.surface,
    #       "time": 0,
    #       "terminal": 0,
    #   }
    #   if get_jax:

    #     return seaquest.EnvState(**state_gym_to_jax)
    elif env_name == "SpaceInvaders-MinAtar":
        state_gym_to_jax = {
            "pos": env.env.pos,
            "f_bullet_map": jnp.array(env.env.f_bullet_map),
            "e_bullet_map": jnp.array(env.env.e_bullet_map),
            "alien_map": jnp.array(env.env.alien_map),
            "alien_dir": env.env.alien_dir,
            "enemy_move_interval": env.env.enemy_move_interval,
            "alien_move_timer": env.env.alien_move_timer,
            "alien_shot_timer": env.env.alien_shot_timer,
            "ramp_index": env.env.ramp_index,
            "shot_timer": env.env.shot_timer,
            "ramping": env.env.ramping,
            "time": 0,
            "terminal": False,
        }
        if get_jax:

            return space_invaders.EnvState(**state_gym_to_jax)
    return state_gym_to_jax

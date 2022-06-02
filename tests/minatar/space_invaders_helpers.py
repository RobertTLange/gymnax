import numpy as np


def get_jax_state_from_numpy(env):
    """A helper for summarizing numpy env info into JAX state."""
    state_jax = {
        "pos": env.env.pos,
        "f_bullet_map": env.env.f_bullet_map,
        "e_bullet_map": env.env.e_bullet_map,
        "alien_map": env.env.alien_map,
        "alien_dir": env.env.alien_dir,
        "enemy_move_interval": env.env.enemy_move_interval,
        "alien_move_timer": env.env.alien_move_timer,
        "alien_shot_timer": env.env.alien_shot_timer,
        "ramp_index": env.env.ramp_index,
        "shot_timer": env.env.shot_timer,
        "terminal": env.env.terminal,
        "ramping": True,
    }
    return state_jax


def step_agent_numpy(env, action):
    """Part of numpy env state transition - Only agent."""
    a = env.env.action_map[action]
    shot_cool_down = 5

    # Resolve player action
    if a == "f" and env.env.shot_timer == 0:
        env.env.f_bullet_map[9, env.env.pos] = 1
        env.env.shot_timer = shot_cool_down
    elif a == "l":
        env.env.pos = max(0, env.env.pos - 1)
    elif a == "r":
        env.env.pos = min(9, env.env.pos + 1)

    # Update Friendly Bullets
    env.env.f_bullet_map = np.roll(env.env.f_bullet_map, -1, axis=0)
    env.env.f_bullet_map[9, :] = 0

    # Update Enemy Bullets
    env.env.e_bullet_map = np.roll(env.env.e_bullet_map, 1, axis=0)
    env.env.e_bullet_map[0, :] = 0
    env.env.terminal = env.env.e_bullet_map[9, env.env.pos]
    return env.env.terminal


def step_aliens_numpy(env):
    """Part of numpy env state transition - Update aliens."""
    terminal_1, terminal_2, terminal_3 = 0, 0, 0
    if env.env.alien_map[9, env.env.pos]:
        terminal_1 = 1
    if env.env.alien_move_timer == 0:
        env.env.alien_move_timer = min(
            np.count_nonzero(env.env.alien_map), env.env.enemy_move_interval
        )
        if (np.sum(env.env.alien_map[:, 0]) > 0 and env.env.alien_dir < 0) or (
            np.sum(env.env.alien_map[:, 9]) > 0 and env.env.alien_dir > 0
        ):
            env.env.alien_dir = -env.env.alien_dir
            if np.sum(env.env.alien_map[9, :]) > 0:
                terminal_2 = 1
            env.env.alien_map = np.roll(env.env.alien_map, 1, axis=0)
        else:
            env.env.alien_map = np.roll(
                env.env.alien_map, env.env.alien_dir, axis=1
            )
        if env.env.alien_map[9, env.env.pos]:
            terminal_3 = 1
    env.env.terminal = (terminal_1 + terminal_2 + terminal_3) > 0
    return env.env.terminal


def step_shoot_numpy(env):
    r = 0
    enemy_shot_interval = 10
    if env.env.alien_shot_timer == 0:
        env.env.alien_shot_timer = enemy_shot_interval
        nearest_alien = env.env._nearest_alien(env.env.pos)
        env.env.e_bullet_map[nearest_alien[0], nearest_alien[1]] = 1

    kill_locations = np.logical_and(
        env.env.alien_map, env.env.alien_map == env.env.f_bullet_map
    )

    r += np.sum(kill_locations)
    env.env.alien_map[kill_locations] = env.env.f_bullet_map[kill_locations] = 0
    return r


def get_nearest_alien_numpy(env):
    """Get closest alien to shoot."""
    search_order = [i for i in range(10)]
    search_order.sort(key=lambda x: abs(x - env.env.pos))
    # Loop over distances and check if there is alien left
    for i in search_order:
        if np.sum(env.env.alien_map[:, i]) > 0:
            return [np.max(np.where(env.env.alien_map[:, i] == 1)), i]
    return None

ramp_interval = 100
init_spawn_speed = 10
init_move_interval = 5
shot_cool_down = 5


def step_agent_numpy(env, action):
    """Numpy helper - Update environment according to agent action."""
    a = env.env.action_map[action]
    # Resolve player action
    if a == "l":
        env.env.player_x = max(0, env.env.player_x - 1)
    elif a == "r":
        env.env.player_x = min(9, env.env.player_x + 1)
    elif a == "u":
        env.env.player_y = max(1, env.env.player_y - 1)
    elif a == "d":
        env.env.player_y = min(8, env.env.player_y + 1)


def step_entities_numpy(env):
    """Numpy helper - Update entities and calculate reward."""
    r = 0
    for i in range(len(env.env.entities)):
        x = env.env.entities[i]
        if x is not None:
            if x[0:2] == [env.env.player_x, env.env.player_y]:
                if env.env.entities[i][3]:
                    env.env.entities[i] = None
                    r += 1
                else:
                    env.env.terminal = True

    if env.env.move_timer == 0:
        env.env.move_timer = env.env.move_speed
        for i in range(len(env.env.entities)):
            x = env.env.entities[i]
            if x is not None:
                x[0] += 1 if x[2] else -1
                if x[0] < 0 or x[0] > 9:
                    env.env.entities[i] = None
                if x[0:2] == [env.env.player_x, env.env.player_y]:
                    if env.env.entities[i][3]:
                        env.env.entities[i] = None
                        r += 1
                    else:
                        env.env.terminal = True
    return r


def step_timers_numpy(env):
    """Numpy help - update timers and ramp difficulty."""
    # Update various timers
    env.env.spawn_timer -= 1
    env.env.move_timer -= 1

    # Ramp difficulty if interval has elapsed
    if env.env.ramping and (env.env.spawn_speed > 1 or env.env.move_speed > 1):
        if env.env.ramp_timer >= 0:
            env.env.ramp_timer -= 1
        else:
            if env.env.move_speed > 1 and env.env.ramp_index % 2:
                env.env.move_speed -= 1
            if env.env.spawn_speed > 1:
                env.env.spawn_speed -= 1
            env.env.ramp_index += 1
            env.env.ramp_timer = ramp_interval

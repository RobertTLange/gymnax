import numpy as np


def step_agent_numpy(env, action):
    """Helper that steps agent pos and ball_x, ball_y."""
    a = env.env.action_map[action]

    # Resolve player action
    if a == "l":
        env.env.pos = max(0, env.env.pos - 1)
    elif a == "r":
        env.env.pos = min(9, env.env.pos + 1)

    # Update ball position
    env.env.last_x = env.env.ball_x
    env.env.last_y = env.env.ball_y
    if env.env.ball_dir == 0:
        new_x = env.env.ball_x - 1
        new_y = env.env.ball_y - 1
    elif env.env.ball_dir == 1:
        new_x = env.env.ball_x + 1
        new_y = env.env.ball_y - 1
    elif env.env.ball_dir == 2:
        new_x = env.env.ball_x + 1
        new_y = env.env.ball_y + 1
    elif env.env.ball_dir == 3:
        new_x = env.env.ball_x - 1
        new_y = env.env.ball_y + 1

    if new_x < 0 or new_x > 9:
        if new_x < 0:
            new_x = 0
        if new_x > 9:
            new_x = 9
        env.env.ball_dir = [1, 0, 3, 2][env.env.ball_dir]
    return new_x, new_y


def step_ball_brick_numpy(env, new_x, new_y):
    """Helper that implements brick map conditions."""
    r = 0
    strike_toggle = False

    if new_y < 0:
        new_y = 0
        env.env.ball_dir = [3, 2, 1, 0][env.env.ball_dir]
    elif env.env.brick_map[new_y, new_x] == 1:
        strike_toggle = True
        if not env.env.strike:
            r += 1
            env.env.strike = True
            env.env.brick_map[new_y, new_x] = 0
            new_y = env.env.last_y
            env.env.ball_dir = [3, 2, 1, 0][env.env.ball_dir]
    elif new_y == 9:
        if np.count_nonzero(env.env.brick_map) == 0:
            env.env.brick_map[1:4, :] = 1
        if env.env.ball_x == env.env.pos:
            env.env.ball_dir = [3, 2, 1, 0][env.env.ball_dir]
            new_y = env.env.last_y
        elif new_x == env.env.pos:
            env.env.ball_dir = [2, 3, 0, 1][env.env.ball_dir]
            new_y = env.env.last_y
        else:
            env.env.terminal = True

    if not strike_toggle:
        env.env.strike = False

    env.env.ball_x = new_x
    env.env.ball_y = new_y
    return r, env.env.terminal

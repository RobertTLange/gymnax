import numpy as np


def det_randomize_cars_numpy(speeds, directions, old_cars, initialize):
    """Helper _randomize_cars(self, initialize) function from numpy."""
    # We have extracted all randomness for testing purposes
    speeds_new = directions * speeds
    if initialize:
        cars = []
        for i in range(8):
            cars += [[0, i + 1, abs(speeds_new[i]), speeds_new[i]]]
        return cars
    else:
        for i in range(8):
            old_cars[i][2:4] = [abs(speeds_new[i]), speeds_new[i]]
        return np.array(old_cars)


def step_cars_numpy(env):
    # Update cars
    for car in env.env.cars:
        if car[0:2] == [4, env.env.pos]:
            env.env.pos = 9
        if car[2] == 0:
            car[2] = abs(car[3])
            car[0] += 1 if car[3] > 0 else -1
            if car[0] < 0:
                car[0] = 9
            elif car[0] > 9:
                car[0] = 0
            if car[0:2] == [4, env.env.pos]:
                env.env.pos = 9
        else:
            car[2] -= 1

    # Update various timers
    env.env.move_timer -= env.env.move_timer > 0
    env.env.terminate_timer -= 1
    if env.env.terminate_timer < 0:
        env.env.terminal = True
    return


def step_agent_numpy(env, action):
    """Helper deterministic part of step transition from numpy."""
    r = 0
    a = env.env.action_map[action]
    if a == "u" and env.env.move_timer == 0:
        env.env.move_timer = 3
        env.env.pos = max(0, env.env.pos - 1)
    elif a == "d" and env.env.move_timer == 0:
        env.env.move_timer = 3
        env.env.pos = min(9, env.env.pos + 1)
    if env.env.pos == 0:
        r += 1
        env.env.pos = 9
    return r

import numpy as np


def init_maze(ax, env, state, params):
    im = ax.imshow(env.occupied_map, cmap="Greys")
    anno_pos = ax.annotate(
        "A",
        fontsize=20,
        xy=(state.pos[1], state.pos[0]),
        xycoords="data",
        xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
    )
    anno_goal = ax.annotate(
        "G",
        fontsize=20,
        xy=(state.goal[1], state.goal[0]),
        xycoords="data",
        xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return anno_pos


def update_maze(im, env, state):
    xy = (state.pos[1], state.pos[0])

    xytext = (state.pos[1] - 0.3, state.pos[0] + 0.25)

    im.set_position((xytext[0], xytext[1]))
    im.xy = (xy[0], xy[1])
    return im

"""Visualization for the circle environment."""

import matplotlib.pyplot as plt
import numpy as np


def init_circle(ax, _, state, params):
    """Initializes the visualization for the circle environment.


    Args:
      ax: The matplotlib axis to draw on.
      state: The state of the environment.
      params: The parameters of the environment.


    Returns:
      A list of matplotlib artists to update during the episode.
    """

    angles = np.linspace(0, np.pi, 100)
    x, y = np.cos(angles), np.sin(angles)
    ax.plot(x, y, color="k")
    plt.axis("scaled")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])

    anno_goal = plt.Circle(
        (state.goal[0], state.goal[1]), radius=params.goal_radius, alpha=0.3
    )
    ax.add_artist(anno_goal)

    anno_agent = plt.Circle(
        (state.pos[0], state.pos[1]), radius=0.05, alpha=1, color="red"
    )
    ax.add_artist(anno_agent)
    return [anno_goal, anno_agent]


def update_circle(im, _, state):
    """Updates the visualization for the circle environment.


    Args:
      im: The list of matplotlib artists to update.
      state: The state of the environment.


    Returns:
      A list of matplotlib artists to update during the episode.
    """
    anno_goal = im[0]
    anno_agent = im[1]
    anno_goal.center = (state.goal[0], state.goal[1])
    anno_agent.center = (state.pos[0], state.pos[1])
    return [anno_goal, anno_agent]

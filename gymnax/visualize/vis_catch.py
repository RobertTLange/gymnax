"""Visualization for Catch."""

import numpy as np


def init_catch(ax, env, state, _):
    """Initialize the visualization for Catch.


    Args:
      ax: The matplotlib axis to draw on.
      env: The environment.
      state: The initial state.
      _: The parameters.


    Returns:
      The annotations for the paddle and ball.
    """

    _ = ax.imshow(np.zeros((env.rows, env.columns)), cmap="Greys", vmin=0, vmax=1)
    anno_paddle = ax.annotate(
        "P",
        fontsize=20,
        xy=(state.paddle_x, state.paddle_y),
        xycoords="data",
        xytext=(state.paddle_x - 0.3, state.paddle_y + 0.25),
    )
    anno_ball = ax.annotate(
        "B",
        fontsize=20,
        xy=(state.ball_x, state.ball_y),
        xycoords="data",
        xytext=(state.ball_x - 0.3, state.ball_y + 0.25),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return (anno_paddle, anno_ball)


def update_catch(im, _, state):
    """Update the visualization for Catch.


    Args:
      im: The annotations for the paddle and ball.
      _: The environment.
      state: The current state.


    Returns:
      The updated annotations for the paddle and ball.
    """

    anno_paddle, anno_ball = im

    xy_p = (state.paddle_x, state.paddle_y)
    xytext_p = (state.paddle_x - 0.3, state.paddle_y + 0.25)

    anno_paddle.set_position((xytext_p[0], xytext_p[1]))
    anno_paddle.xy = (xy_p[0], xy_p[1])

    xy_b = (state.ball_x, state.ball_y)
    xytext_b = (state.ball_x - 0.3, state.ball_y + 0.25)

    anno_ball.set_position((xytext_b[0], xytext_b[1]))
    anno_ball.xy = (xy_b[0], xy_b[1])
    return (anno_paddle, anno_ball)

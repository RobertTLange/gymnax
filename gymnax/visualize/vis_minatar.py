"""Visualize Minatar environments."""

from matplotlib import colors
import numpy as np
import seaborn as sns


def init_minatar(ax, env, state):
    """Initialize the Minatar visualization."""

    obs = env.get_obs(state)
    n_channels = env.obs_shape[-1]
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    numerical_state = (
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation="none")


def update_minatar(im, env, state):
    """Update the Minatar visualization."""
    obs = env.get_obs(state)
    n_channels = env.obs_shape[-1]
    numerical_state = (
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
    )
    im.set_data(numerical_state)

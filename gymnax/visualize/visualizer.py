import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from .vis_minatar import init_minatar, update_minatar


class Visualizer(object):
    def __init__(self, env, env_params, state_seq: list):
        self.env = env
        self.env_params = env_params
        self.state_seq = state_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))
        self.interval = 150

    def animate(
        self,
        save_fname: Optional[str] = "test.gif",
        view: bool = False,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)
        # Simply view it 3 times
        if view:
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    def init(self):
        # Plot placeholder points
        if self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
        ]:
            self.im = init_minatar(self.ax, self.env, self.state_seq[0])
        self.fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    def update(self, frame):
        if self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
        ]:
            update_minatar(self.im, self.env, self.state_seq[frame])
        self.ax.set_title(f"{self.env.name} - Time Step {frame + 1}")

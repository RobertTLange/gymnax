import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from .vis_minatar import init_minatar, update_minatar
from .vis_circle import init_circle, update_circle
from .vis_maze import init_maze, update_maze


class Visualizer(object):
    def __init__(self, env, env_params, state_seq: list):
        self.env = env
        self.env_params = env_params
        self.state_seq = state_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))
        self.interval = 500

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
        elif self.env.name == "PointRobot-misc":
            self.im = init_circle(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        elif self.env.name == "MetaMaze-misc":
            self.im = init_maze(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
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
        elif self.env.name == "PointRobot-misc":
            self.im = update_circle(self.im, self.env, self.state_seq[frame])
        elif self.env.name == "MetaMaze-misc":
            self.im = update_maze(self.im, self.env, self.state_seq[frame])
        self.ax.set_title(f"{self.env.name} - Time Step {frame + 1}")

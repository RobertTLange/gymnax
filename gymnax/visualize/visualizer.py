import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from .vis_gym import init_gym, update_gym
from .vis_minatar import init_minatar, update_minatar
from .vis_circle import init_circle, update_circle
from .vis_maze import init_maze, update_maze
from .vis_catch import init_catch, update_catch


class Visualizer(object):
    def __init__(self, env, env_params, state_seq, reward_seq=None):
        self.env = env
        self.env_params = env_params
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))
        if env.name not in [
            "Acrobot-v1",
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ]:
            self.interval = 100
        else:
            self.interval = 50

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
            "Acrobot-v1",
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ]:
            import gym

            # Animations have to use older gym version and pyglet!
            assert gym.__version__ == "0.19.0"
            self.im = init_gym(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        elif self.env.name == "Catch-bsuite":
            self.im = init_catch(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        elif self.env.name in [
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
        elif self.env.name in ["MetaMaze-misc", "FourRooms-misc"]:
            self.im = init_maze(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        self.fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    def update(self, frame):
        if self.env.name in [
            "Acrobot-v1",
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ]:
            self.im = update_gym(self.im, self.env, self.state_seq[frame])
        elif self.env.name == "Catch-bsuite":
            self.im = update_catch(self.im, self.env, self.state_seq[frame])
        elif self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
        ]:
            update_minatar(self.im, self.env, self.state_seq[frame])
        elif self.env.name == "PointRobot-misc":
            self.im = update_circle(self.im, self.env, self.state_seq[frame])
        elif self.env.name in ["MetaMaze-misc", "FourRooms-misc"]:
            self.im = update_maze(self.im, self.env, self.state_seq[frame])

        if self.reward_seq is None:
            self.ax.set_title(
                f"{self.env.name} - Step {frame + 1}", fontsize=15
            )
        else:
            self.ax.set_title(
                "{}: Step {:4.0f} - Return {:7.2f}".format(
                    self.env.name, frame + 1, self.reward_seq[frame]
                ),
                fontsize=15,
            )

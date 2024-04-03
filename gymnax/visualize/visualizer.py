"""Visualizer for Gymnax environments."""

from typing import Optional
import gym
import jax
import jax.numpy as jnp
from matplotlib import animation
import matplotlib.pyplot as plt
import gymnax
from gymnax.visualize import vis_catch
from gymnax.visualize import vis_circle
from gymnax.visualize import vis_gym
from gymnax.visualize import vis_maze
from gymnax.visualize import vis_minatar


class Visualizer(object):
    """Visualizer for Gymnax environments."""

    def __init__(self, env_arg, env_params_arg, state_seq_arg, reward_seq_arg=None):

        self.env = env_arg
        self.env_params = env_params_arg
        self.state_seq = state_seq_arg
        self.reward_seq = reward_seq_arg
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))
        if env_arg.name not in [
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
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)."""
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
        """Plot placeholder points."""
        if self.env.name in [
            "Acrobot-v1",
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ]:

            # Animations have to use older gym version and pyglet!
            assert gym.__version__ == "0.19.0"
            self.im = vis_gym.init_gym(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        elif self.env.name == "Catch-bsuite":
            self.im = vis_catch.init_catch(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        elif self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
            "Pong-misc",
        ]:
            self.im = vis_minatar.init_minatar(self.ax, self.env, self.state_seq[0])
        elif self.env.name == "PointRobot-misc":
            self.im = vis_circle.init_circle(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        elif self.env.name in ["MetaMaze-misc", "FourRooms-misc"]:
            self.im = vis_maze.init_maze(
                self.ax, self.env, self.state_seq[0], self.env_params
            )
        self.fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    def update(self, frame):
        """Update the animation."""
        if self.env.name in [
            "Acrobot-v1",
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ]:
            self.im = vis_gym.update_gym(self.im, self.env, self.state_seq[frame])
        elif self.env.name == "Catch-bsuite":
            self.im = vis_catch.update_catch(self.im, self.env, self.state_seq[frame])
        elif self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
            "Pong-misc",
        ]:
            vis_minatar.update_minatar(self.im, self.env, self.state_seq[frame])
        elif self.env.name == "PointRobot-misc":
            self.im = vis_circle.update_circle(self.im, self.env, self.state_seq[frame])
        elif self.env.name in ["MetaMaze-misc", "FourRooms-misc"]:
            self.im = vis_maze.update_maze(self.im, self.env, self.state_seq[frame])

        if self.reward_seq is None:
            self.ax.set_title(f"{self.env.name} - Step {frame + 1}", fontsize=15)
        else:
            self.ax.set_title(
                "{}: Step {:4.0f} - Return {:7.2f}".format(
                    self.env.name, frame + 1, self.reward_seq[frame]
                ),
                fontsize=15,
            )


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("Pong-misc")

    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate("anim.gif")

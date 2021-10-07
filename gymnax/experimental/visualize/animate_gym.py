from matplotlib import animation
import matplotlib.pyplot as plt

try:
    import gym
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        f"{err}. You need to install `gym` "
        "to use the `evosax.visualize.animate_gym` "
        "module."
    )


class GymAnimator:
    def __init__(self, policy, policy_params, env_name):
        """Init policy fct., params and environment."""
        self.policy = policy
        self.policy_params = policy_params
        self.env = gym.make(env_name)

    def collect_frames(self, num_steps):
        """Rollout a single episode using provided policy + params."""
        state = self.env.reset()
        frames = []
        total_reward = 0
        for t in range(num_steps):
            # Render to frames buffer
            frame = self.env.render(mode="rgb_array")
            frames.append(frame)
            # We assume that policy directly outputs action
            # to be exectued in the environment
            action = self.policy(self.policy_params, state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        self.env.close()
        return frames, total_reward

    def animate(self, frames, title, reward, filename):
        """Rollout an episode given the provided policy and visualize it."""
        self.save_frames_as_gif(frames, title=title, filename=filename, reward=reward)
        print("Finished processing frames to .gif.")

    def save_frames_as_gif(self, frames, title, filename="gym_animation.gif", reward=0):
        """Animate a set of collected episode frames."""
        plt.figure(
            figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72
        )

        patch = plt.imshow(frames[0])
        plt.title(title + r" | R: {:.1f}".format(reward), fontsize=50)
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(frames), interval=50
        )
        anim.save(filename, writer="imagemagick", fps=30)

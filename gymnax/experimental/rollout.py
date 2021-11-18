import jax
import jax.numpy as jnp
import gymnax


class EnvRollout(object):
    def __init__(
        self,
        model_forward,
        env_name: str = "Pendulum-v1",
        num_env_steps: int = 200,
        num_episodes: int = 20,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env_name = env_name
        self.num_env_steps = num_env_steps
        self.num_episodes = num_episodes

        # Define the RL environment & network forward function
        self.env, self.env_params = gymnax.make(self.env_name)
        self.model_forward = model_forward

        # Set up the generation evaluation vmap-ed function - rl/supervised/etc.
        self.gen_evaluate = self.batch_evaluate()

    def collect(self, rng_eval, policy_params):
        """Reshape parameter vector and evaluate the generation."""
        # Reshape the parameters into the correct network format
        rollout_keys = jax.random.split(rng_eval, self.num_episodes)

        # Evaluate generation population on pendulum task - min cost!
        pop_trajectories = self.gen_evaluate(rollout_keys, policy_params)
        return pop_trajectories

    def batch_evaluate(self):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.jit(
            jax.vmap(self.rollout, in_axes=(0, None), out_axes=0)
        )
        return batch_rollout

    def rollout(self, rng_input, policy_params):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.model_forward({"params": policy_params}, obs, rng=rng_net)
            next_o, next_s, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            carry = [next_o.squeeze(), next_s, policy_params, rng]
            y = [next_o.squeeze(), reward, done]
            return carry, y

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [obs, state, policy_params, rng_episode],
            [jnp.zeros((self.num_env_steps, self.input_shape[0] + 2))],
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, rewards, dones = scan_out[0], scan_out[1], scan_out[2]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return obs, rewards, dones, jnp.sum(rewards * ep_mask)

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape

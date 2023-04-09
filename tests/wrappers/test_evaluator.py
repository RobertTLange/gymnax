import jax
import jax.numpy as jnp
import flax.linen as nn
from gymnax.experimental import RolloutWrapper


class MLP(nn.Module):
    """Simple MLP Wrapper with flexible output head."""

    @nn.compact
    def __call__(self, x, rng):
        # Loop over dense layers in forward pass
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        x = nn.tanh(x)
        return x


def test_rollout():
    rng = jax.random.PRNGKey(0)
    model = MLP()
    pholder = jnp.zeros((3,))
    policy_params = model.init(
        rng,
        x=pholder,
        rng=rng,
    )
    manager = RolloutWrapper(
        model.apply, env_name="Pendulum-v1", num_env_steps=200
    )

    # Test simple single episode rollout
    obs, action, reward, next_obs, done, cum_return = manager.single_rollout(
        rng, policy_params
    )
    assert obs.shape == (200, 3)

    # Test multiple rollouts for same network (different random numbers)
    rng_batch = jax.random.split(rng, 10)
    obs, action, reward, next_obs, done, cum_return = manager.batch_rollout(
        rng_batch, policy_params
    )
    assert obs.shape == (10, 200, 3)

    # Test multiple rollouts for different networks
    batch_params = jax.tree_map(
        lambda x: jnp.tile(x, (5, 1)).reshape(5, *x.shape), policy_params
    )
    # print(jax.tree_map(lambda x: x.shape, policy_params))
    # print(jax.tree_map(lambda x: x.shape, batch_params))
    (
        obs,
        action,
        reward,
        next_obs,
        done,
        cum_return,
    ) = manager.population_rollout(rng_batch, batch_params)
    assert obs.shape == (5, 10, 200, 3)

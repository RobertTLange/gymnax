import gymnax
import jax
import jax.numpy as jnp
from itertools import product
from bsuite.environments.deep_sea import DeepSea
import numpy as np
import jax.random as jr


def generate_all_action_sequences(size):
    """Generate all possible action sequences for DeepSea environment.

    Each episode lasts exactly `size` steps, with binary actions (0 or 1).
    Returns all 2^size possible sequences.
    """
    return list(product([0, 1], repeat=size))


def run_episode_original(env, action_sequence):
    """Run a complete episode on the original bsuite environment."""
    total_reward = 0.0
    timestep = env.reset()

    for action in action_sequence:
        timestep = env.step(action)
        total_reward += timestep.reward
    assert timestep.last()

    return total_reward


def run_episode_gymnax(env, env_params, action_sequence, key):
    """Run a complete episode on the gymnax environment."""
    obs, state = env.reset(key, env_params)
    total_reward = 0.0

    for action in action_sequence:
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = env.step(subkey, state, action, env_params)
        total_reward += reward

    assert done
    return float(total_reward)


def deepsea_equivalence(env_config):
    """Test that both DeepSea implementations produce identical rewards for all action sequences."""

    # Initialize both environments with same seeds
    original_env = DeepSea(
        **env_config,
        seed=0,
        mapping_seed=0,
    )

    our_env, env_params = gymnax.make(
        "DeepSea-bsuite", **env_config, action_mapping_rng_key=jr.key(0)
    )
    # TODO: generalize this by making a monkey-patched generators for jax/numpy/random etc
    if ("randomize_actions" not in env_config) or env_config["randomize_actions"]:
        our_env.fixed_action_mapping = jnp.asarray(original_env._action_mapping).astype(
            jnp.bool_
        )
    key = jr.key(0)

    # Generate all possible action sequences for the environment size
    all_sequences = generate_all_action_sequences(env_config["size"])

    mismatches = []

    for i, action_sequence in enumerate(all_sequences):
        # Run episode on original environment
        original_reward = run_episode_original(original_env, action_sequence)
        assert (our_env.fixed_action_mapping == original_env._action_mapping).all()

        # Run episode on gymnax environment
        key, subkey = jax.random.split(key)
        gymnax_reward = run_episode_gymnax(our_env, env_params, action_sequence, subkey)

        # Compare rewards with small tolerance for floating point differences
        if not np.isclose(original_reward, gymnax_reward, rtol=1e-6, atol=1e-6):
            mismatches.append(
                {
                    "sequence": action_sequence,
                    "original_reward": original_reward,
                    "gymnax_reward": gymnax_reward,
                    "difference": abs(original_reward - gymnax_reward),
                }
            )

    # Report results
    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches!")
        for mismatch in mismatches[:10]:  # Show first 10 mismatches
            print(f"Sequence: {mismatch['sequence']}")
            print(f"  Original: {mismatch['original_reward']}")
            print(f"  Gymnax:   {mismatch['gymnax_reward']}")
            print(f"  Diff:     {mismatch['difference']}")

        if len(mismatches) > 10:
            print(f"... and {len(mismatches) - 10} more mismatches")

        assert False, f"Reward mismatches found in {len(mismatches)} sequences"


def test_deepsea_equivalence():
    for size in [4, 8]:
        for deterministic in [True]:
            for unscaled_move_cost in [0, 0.01]:
                for randomize_actions in [True, False]:
                    env_config = {
                        "size": size,
                        "deterministic": deterministic,
                        "unscaled_move_cost": unscaled_move_cost,
                        "randomize_actions": randomize_actions,
                    }
                    deepsea_equivalence(env_config)
    # test the default parameters
    env_config = {"size": 12}
    deepsea_equivalence(env_config)


if __name__ == "__main__":
    test_deepsea_equivalence()

"""Fixed-buffer array operations shared by Seaquest transitions and rendering."""

import jax
import jax.numpy as jnp

BOARD_SIZE = 10
MAX_ENTITIES = 100


def active_mask(count: jax.Array) -> jax.Array:
    """Return a fixed-size mask for the active prefix of an entity buffer."""
    return jnp.arange(MAX_ENTITIES) < count


def compact(entities: jax.Array, keep: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Remove entity rows while retaining their source-list ordering."""
    destinations = jnp.maximum(jnp.cumsum(keep.astype(jnp.int32)) - 1, 0)
    values = jnp.where(keep[:, None], entities, 0)
    compacted = jnp.zeros_like(entities).at[destinations].add(values)
    return compacted, jnp.sum(keep, dtype=jnp.int32)


def append(
    entities: jax.Array, count: jax.Array, entity: jax.Array, enabled: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Append one entity when fixed capacity remains."""
    can_append = jnp.logical_and(enabled, count < MAX_ENTITIES)
    safe_index = jnp.minimum(count, MAX_ENTITIES - 1)
    updated = jnp.where(can_append, entity, entities[safe_index])
    return entities.at[safe_index].set(updated), count + can_append.astype(jnp.int32)


def append_many(
    entities: jax.Array, count: jax.Array, new_entities: jax.Array, enabled: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Append a bounded set of entities in row order."""
    available = jnp.maximum(MAX_ENTITIES - count, 0)
    ranks = jnp.cumsum(enabled.astype(jnp.int32)) - 1
    add_mask = jnp.logical_and(enabled, ranks < available)
    destinations = jnp.minimum(count + jnp.maximum(ranks, 0), MAX_ENTITIES - 1)
    values = jnp.where(add_mask[:, None], new_entities, 0)
    return entities.at[destinations].add(values), count + jnp.sum(
        add_mask, dtype=jnp.int32
    )


def move_entities(
    entities: jax.Array, count: jax.Array, move_speed: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Advance fish, subs, or divers when their movement timer reaches zero."""
    active = active_mask(count)
    ready = jnp.logical_and(active, entities[:, 3] == 0)
    direction = jnp.where(entities[:, 2] != 0, 1, -1)
    next_x = jnp.where(ready, entities[:, 0] + direction, entities[:, 0])
    next_timer = jnp.where(ready, move_speed, jnp.maximum(entities[:, 3] - 1, 0))
    entities = entities.at[:, 0].set(next_x).at[:, 3].set(next_timer)
    in_bounds = (next_x >= 0) & (next_x < BOARD_SIZE)
    return compact(entities, active & in_bounds)


def move_bullets(entities: jax.Array, count: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Move bullets and discard bullets leaving the board."""
    active = active_mask(count)
    next_x = entities[:, 0] + jnp.where(entities[:, 2] != 0, 1, -1)
    entities = entities.at[:, 0].set(next_x)
    return compact(entities, active & (next_x >= 0) & (next_x < BOARD_SIZE))


def collide_bullets(
    bullets: jax.Array,
    bullet_count: jax.Array,
    targets: jax.Array,
    target_count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Remove colliding bullets and targets, returning their total reward."""
    bullet_active = active_mask(bullet_count)
    target_active = active_mask(target_count)
    same_position = jnp.all(bullets[:, None, :2] == targets[None, :, :2], axis=-1)
    matches = same_position & bullet_active[:, None] & target_active[None, :]
    first_target = jnp.argmax(matches, axis=1)
    bullet_has_target = jnp.any(matches, axis=1)
    selected_matches = matches & (
        jnp.arange(MAX_ENTITIES)[None, :] == first_target[:, None]
    )
    target_winner = jnp.max(
        jnp.where(selected_matches, jnp.arange(MAX_ENTITIES)[:, None], -1), axis=0
    )
    target_hit = target_winner >= 0
    bullet_hit = bullet_has_target & (
        jnp.arange(MAX_ENTITIES) == target_winner[first_target]
    )
    bullets, bullet_count = compact(bullets, bullet_active & ~bullet_hit)
    targets, target_count = compact(targets, target_active & ~target_hit)
    return (
        bullets,
        bullet_count,
        targets,
        target_count,
        jnp.sum(target_hit, dtype=jnp.float32),
    )


def draw_entities(
    observation: jax.Array,
    entities: jax.Array,
    count: jax.Array,
    channel: int,
    trail: bool,
) -> jax.Array:
    """Draw active entity positions and optional direction trails."""
    active = active_mask(count)
    x = jnp.clip(entities[:, 0], 0, BOARD_SIZE - 1)
    y = jnp.clip(entities[:, 1], 0, BOARD_SIZE - 1)
    observation = observation.at[y, x, channel].max(active.astype(jnp.float32))
    if not trail:
        return observation
    back_x = entities[:, 0] + jnp.where(entities[:, 2] != 0, -1, 1)
    in_bounds = (back_x >= 0) & (back_x < BOARD_SIZE)
    return observation.at[y, jnp.clip(back_x, 0, BOARD_SIZE - 1), 3].max(
        (active & in_bounds).astype(jnp.float32)
    )

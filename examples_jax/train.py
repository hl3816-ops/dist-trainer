"""Data-parallel training in JAX via jax.pmap, as a cross-framework
counterpart to the PyTorch DDP work in the rest of this repo -- the JD this
project targets explicitly lists "PyTorch, JAX" as the frameworks to know.

JAX's model is different enough from PyTorch's that it's worth spelling
out: there's no Trainer object wrapping a stateful nn.Module. Parameters
are an explicit pytree, `jax.pmap` compiles the training step once and
runs it in parallel across N *devices within one process* (real
accelerators, or -- as used here -- simulated CPU devices via
XLA_FLAGS=--xla_force_host_platform_device_count=N, so this needs no GPU
to demonstrate the actual multi-device data-parallel mechanics), and
`jax.lax.pmean` inside the pmapped function is JAX's equivalent of DDP's
gradient all-reduce: every device computes gradients on its own shard of
the batch, pmean averages them across devices, and every device applies
the identical averaged update -- so parameters stay in sync without any
external process group, launcher, or NCCL involved at all.
"""

from __future__ import annotations

import os

# Must be set before `import jax` to take effect -- simulates N CPU devices
# so pmap's multi-device code path is exercised without real accelerators.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
from jax import tree_util
from model import init_params, init_target_fn, mse_loss

LR = 1e-2


def make_sharded_batch(
    key: jax.Array, n_devices: int, per_device_batch: int, target_fn
):
    """Shape (n_devices, per_device_batch, features) -- pmap maps axis 0
    across devices automatically. Every device gets a disjoint slice of
    the same virtual full batch for this key, matching the sharding
    convention used throughout the PyTorch side of this repo."""
    total = n_devices * per_device_batch
    kx, kn = jax.random.split(key)
    x = jax.random.normal(kx, (total, 20))
    y = target_fn(x) + 0.05 * jax.random.normal(kn, (total, 1))
    return x.reshape(n_devices, per_device_batch, 20), y.reshape(
        n_devices, per_device_batch, 1
    )


def _sgd_step(params, x, y, pmean_axis: str | None):
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
    if pmean_axis is not None:
        grads = jax.lax.pmean(grads, axis_name=pmean_axis)
        loss = jax.lax.pmean(loss, axis_name=pmean_axis)
    new_params = tree_util.tree_map(lambda p, g: p - LR * g, params, grads)
    return new_params, loss


def train_single_device(seed: int, steps: int, batch_size: int) -> dict:
    key = jax.random.PRNGKey(seed)
    pkey, tkey, dkey = jax.random.split(key, 3)
    params = init_params(pkey)
    target_fn = init_target_fn(tkey)

    for step in range(steps):
        step_key = jax.random.fold_in(dkey, step)
        x, y = make_sharded_batch(
            step_key, n_devices=1, per_device_batch=batch_size, target_fn=target_fn
        )
        params, _loss = _sgd_step(params, x[0], y[0], pmean_axis=None)
    return params


def train_pmap(seed: int, steps: int, n_devices: int, per_device_batch: int) -> dict:
    key = jax.random.PRNGKey(seed)
    pkey, tkey, dkey = jax.random.split(key, 3)
    params = init_params(pkey)
    target_fn = init_target_fn(tkey)

    # Stack n_devices identical copies along a new leading axis -- pmap
    # shards any array by its leading axis across devices, so a "replicated"
    # array is just one whose leading-axis slices all happen to be equal.
    # (jax.device_put_replicated did this directly but is deprecated as of
    # this jax version in favor of manual replication.)
    replicated_params = tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), params)
    p_step = jax.pmap(
        lambda p, x, y: _sgd_step(p, x, y, pmean_axis="devices"), axis_name="devices"
    )

    for step in range(steps):
        step_key = jax.random.fold_in(dkey, step)
        x, y = make_sharded_batch(step_key, n_devices, per_device_batch, target_fn)
        replicated_params, _losses = p_step(replicated_params, x, y)

    # every device's copy is identical after pmean-synced updates; take device 0's
    return tree_util.tree_map(lambda v: v[0], replicated_params)


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}")
    single = train_single_device(seed=0, steps=50, batch_size=32)
    multi = train_pmap(seed=0, steps=50, n_devices=4, per_device_batch=8)

    max_diff = max(float(jnp.max(jnp.abs(single[k] - multi[k]))) for k in single)
    print(f"max abs param diff, single-device vs 4-device pmap: {max_diff:.2e}")

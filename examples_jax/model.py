"""A tiny functional MLP in raw JAX (no flax/haiku), and the same synthetic
regression task used by examples/task.py in the PyTorch half of this repo --
same shapes, same idea (fixed random target function + noise), so the JAX
and PyTorch correctness demos are directly comparable in spirit even though
JAX's functional/pytree style makes the actual code look quite different
from PyTorch's object-oriented nn.Module style.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

IN_FEATURES = 20
HIDDEN = 64


def init_params(key: jax.Array) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (IN_FEATURES, HIDDEN)) * 0.1,
        "b1": jnp.zeros(HIDDEN),
        "w2": jax.random.normal(k2, (HIDDEN, HIDDEN)) * 0.1,
        "b2": jnp.zeros(HIDDEN),
        "w3": jax.random.normal(k3, (HIDDEN, 1)) * 0.1,
        "b3": jnp.zeros(1),
    }


def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    h = jax.nn.relu(x @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def init_target_fn(key: jax.Array):
    """A fixed (never trained) small MLP used to generate learnable
    synthetic targets, same role as examples/task.py's _get_target_fn."""
    k1, k2 = jax.random.split(key, 2)
    target_params = {
        "w1": jax.random.uniform(k1, (IN_FEATURES, HIDDEN), minval=-1, maxval=1),
        "w2": jax.random.uniform(k2, (HIDDEN, 1), minval=-1, maxval=1),
    }

    def target_fn(x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.tanh(x @ target_params["w1"])
        return h @ target_params["w2"]

    return target_fn


def make_batch(key: jax.Array, batch_size: int, target_fn, noise_std: float = 0.05):
    kx, kn = jax.random.split(key)
    x = jax.random.normal(kx, (batch_size, IN_FEATURES))
    y = target_fn(x) + noise_std * jax.random.normal(kn, (batch_size, 1))
    return x, y


def mse_loss(params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)

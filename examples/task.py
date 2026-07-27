"""A tiny synthetic regression task used to exercise the Trainer.

The target function is a fixed random MLP (seeded, never trained) applied to
Gaussian inputs plus a little noise -- learnable signal, no dataset download
needed, and fast enough to train many steps on CPU in seconds.

batch_iter is written so that world_size ranks each training on batch_size
examples produce, after DDP-averaged gradient sync, the *same* update a
single process training on batch_size * world_size examples would produce:
each rank's batch is a deterministic, non-overlapping shard of the same
step's "virtual" full batch, keyed by (step, rank).
"""

from __future__ import annotations

import torch
from torch import nn

IN_FEATURES = 20
HIDDEN = 64


class RegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_FEATURES, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x)


_target_fn = None


def _get_target_fn(seed: int = 1234) -> nn.Module:
    global _target_fn
    if _target_fn is None:
        gen = torch.Generator().manual_seed(seed)
        fn = nn.Sequential(
            nn.Linear(IN_FEATURES, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )
        with torch.no_grad():
            for p in fn.parameters():
                p.copy_(torch.empty_like(p).uniform_(-1, 1, generator=gen))
        fn.eval()
        _target_fn = fn
    return _target_fn


def make_batch_iter(total_batch_size: int, seed: int = 0, noise_std: float = 0.05):
    """Returns a batch_iter(step, rank, world_size) -> (x, y) callable.

    Every rank regenerates the *same* full virtual batch (same seed + step),
    then slices out its own non-overlapping shard. This means a single
    process run (world_size=1) and an N-process run see, in aggregate, the
    exact same per-step data -- the precondition for the DDP-vs-single-
    process correctness check in tests/test_correctness.py.
    """

    def batch_iter(step: int, rank: int, world_size: int):
        assert total_batch_size % world_size == 0, (
            f"total_batch_size={total_batch_size} must be divisible by world_size={world_size}"
        )
        shard_size = total_batch_size // world_size
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        x_full = torch.randn(total_batch_size, IN_FEATURES, generator=gen)
        with torch.no_grad():
            y_full = _get_target_fn(seed)(x_full)
        y_full = y_full + noise_std * torch.randn(y_full.shape, generator=gen)
        start = rank * shard_size
        return x_full[start : start + shard_size], y_full[start : start + shard_size]

    return batch_iter


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)

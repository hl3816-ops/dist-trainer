"""Character-level language modeling data for TinyGPT, using the tiny-
shakespeare text (~1MB, public domain) -- real text, not synthetic, so the
benchmark story is "trained on real data" not "trained on random noise".

Downloads and caches the text on first use. Falls back to a small bundled
excerpt if there's no internet access (e.g. a Kaggle kernel with internet
disabled), so the benchmark can still run, just on less data.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

_FALLBACK_TEXT = (
    """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.
"""
    * 200
)  # repeat so there's enough length for a few hundred training steps


def load_text(cache_path: str | Path = ".cache/tinyshakespeare.txt") -> str:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - any network failure -> offline fallback
        text = _FALLBACK_TEXT

    cache_path.write_text(text, encoding="utf-8")
    return text


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[i] for i in ids.tolist())


def make_batch_iter(
    data: torch.Tensor, block_size: int, total_batch_size: int, seed: int = 0
):
    """Returns batch_iter(step, rank, world_size) -> (x, y) for next-token
    prediction, sharded the same way as examples/task.py's make_batch_iter:
    every rank draws the same full virtual batch of starting positions for a
    given step, then takes its own shard -- so single-process and N-process
    runs see equivalent aggregate data per step."""

    n = data.shape[0]

    def batch_iter(step: int, rank: int, world_size: int):
        assert total_batch_size % world_size == 0
        shard_size = total_batch_size // world_size
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        starts = torch.randint(
            0, n - block_size - 1, (total_batch_size,), generator=gen
        )
        start = rank * shard_size
        my_starts = starts[start : start + shard_size]
        x = torch.stack([data[s : s + block_size] for s in my_starts])
        y = torch.stack([data[s + 1 : s + 1 + block_size] for s in my_starts])
        return x, y

    return batch_iter

"""Checkpoint save/load for fault-tolerant training.

Only rank 0 writes the checkpoint file (all ranks have identical model state
after a DDP gradient sync, so writing from every rank would just be redundant
disk I/O and a race on the same path). Every rank loads the same file when
resuming, so all ranks come back into an identical state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class CheckpointState:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    torch_rng_state: torch.Tensor
    numpy_rng_state: dict
    python_rng_state: tuple


def save_checkpoint(
    path: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
) -> None:
    """Only rank 0 actually writes to disk; other ranks no-op."""
    if rank != 0:
        return
    # If model is DDP-wrapped, .module gives the underlying model whose
    # state_dict doesn't have the "module." prefix, so it loads cleanly
    # whether or not the loader is also DDP-wrapped.
    underlying = model.module if hasattr(model, "module") else model
    state = {
        "step": step,
        "model_state": underlying.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)  # atomic on POSIX and Windows -- avoids a torn
    # checkpoint file if the process is killed mid-write


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> int:
    """Loads state into model (and optimizer, if given) in place. Returns the
    step number the checkpoint was saved at, so training can resume from
    step + 1 rather than restarting the step counter."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    underlying = model.module if hasattr(model, "module") else model
    underlying.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
    return state["step"]

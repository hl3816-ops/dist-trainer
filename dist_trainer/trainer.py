"""Reusable distributed-training trainer.

Domain-agnostic: takes any nn.Module + optimizer, wraps in DDP when launched
under a distributed launcher (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK), and
otherwise runs as plain single-process training with the exact same code
path -- callers don't need an if/else for distributed vs not.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dist_trainer.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
    ):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.step = 0
        self._is_setup = False

    def setup(self, backend: str = "gloo") -> None:
        if self.distributed:
            dist.init_process_group(backend=backend)
            self.model = DDP(self.model)
        self._is_setup = True

    def teardown(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        assert self._is_setup, "call trainer.setup() before training"
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def maybe_resume(self, checkpoint_dir: str | Path) -> int:
        """Loads the latest checkpoint if one exists. All ranks call this and
        read the same file, so every rank comes back into identical state.
        Returns the step training should resume from (0 if no checkpoint)."""
        path = Path(checkpoint_dir) / "latest.pt"
        if path.exists():
            self.step = load_checkpoint(path, self.model, self.optimizer)
        return self.step

    def maybe_checkpoint(self, checkpoint_dir: str | Path, every: int | None) -> None:
        if not checkpoint_dir or not every or self.step % every != 0:
            return
        save_checkpoint(
            Path(checkpoint_dir) / "latest.pt",
            self.step,
            self.model,
            self.optimizer,
            self.rank,
        )
        if self.distributed:
            # block every rank here so nobody races ahead and tries to load
            # a checkpoint rank 0 hasn't finished writing yet
            dist.barrier()

    def fit(
        self,
        batch_iter,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        resume: bool = True,
    ) -> list[float]:
        if resume and checkpoint_dir is not None:
            self.maybe_resume(checkpoint_dir)

        losses = []
        while self.step < num_steps:
            batch = batch_iter(self.step, self.rank, self.world_size)
            loss = self.train_step(batch, loss_fn)
            losses.append(loss)
            self.maybe_checkpoint(checkpoint_dir, checkpoint_every)
        return losses

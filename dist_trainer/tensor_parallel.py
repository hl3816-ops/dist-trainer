"""Tensor parallelism: split individual layers across ranks, instead of
replicating the whole model (DDP) or sharding it but still running each
layer whole on one rank at a time (FSDP). This is the Megatron-LM technique
for models too big -- or too latency-sensitive -- for data parallelism
alone; DDP/FSDP still need a full layer's activations to fit on one GPU to
compute it, tensor parallelism doesn't.

Implements the standard Megatron-LM MLP pattern: a ColumnParallelLinear
(splits its OUTPUT features across ranks, no communication needed) feeding
a RowParallelLinear (splits its INPUT features to match, then all-reduces
its partial sums together) -- exactly one collective per MLP block, not one
per layer, because the column/row split is chosen specifically so the
intermediate activation between them never needs to be gathered.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class ColumnParallelLinear(nn.Module):
    """Splits the weight matrix by OUTPUT feature across ranks: each rank
    computes a disjoint slice of the output, using its full input. No
    communication in forward -- the output is left sharded, meant to feed
    directly into a RowParallelLinear."""

    def __init__(
        self, in_features: int, out_features: int, world_size: int, bias: bool = True
    ):
        super().__init__()
        assert out_features % world_size == 0, (
            f"out_features={out_features} must be divisible by world_size={world_size}"
        )
        self.out_features_per_rank = out_features // world_size
        self.linear = nn.Linear(in_features, self.out_features_per_rank, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Splits the weight matrix by INPUT feature across ranks: each rank
    computes a partial sum from its shard of the input (the sharded output
    of a preceding ColumnParallelLinear), then all-reduces those partial
    sums together into the true full output. Bias is added once, after the
    reduce, identically on every rank (adding it before would sum it
    world_size times)."""

    def __init__(
        self, in_features: int, out_features: int, world_size: int, bias: bool = True
    ):
        super().__init__()
        assert in_features % world_size == 0, (
            f"in_features={in_features} must be divisible by world_size={world_size}"
        )
        self.world_size = world_size
        self.linear = nn.Linear(in_features // world_size, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        local_out = self.linear(x_shard)
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            local_out = local_out + self.bias
        return local_out


class TensorParallelMLP(nn.Module):
    """A transformer MLP block (Linear -> GELU -> Linear) split across
    ranks with exactly one all-reduce per forward pass, regardless of
    world_size, via the column-then-row pattern above."""

    def __init__(self, n_embd: int, world_size: int, hidden_mult: int = 4):
        super().__init__()
        self.fc = ColumnParallelLinear(n_embd, hidden_mult * n_embd, world_size)
        self.proj = RowParallelLinear(hidden_mult * n_embd, n_embd, world_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(nn.functional.gelu(self.fc(x)))


def load_shard_from_reference(
    tp_mlp: TensorParallelMLP,
    reference_fc: nn.Linear,
    reference_proj: nn.Linear,
    rank: int,
) -> None:
    """Copies this rank's shard of a full-size reference MLP's weights into
    a TensorParallelMLP, for correctness testing against the reference."""
    out_per_rank = tp_mlp.fc.out_features_per_rank
    fc_slice = slice(rank * out_per_rank, (rank + 1) * out_per_rank)
    with torch.no_grad():
        tp_mlp.fc.linear.weight.copy_(reference_fc.weight[fc_slice, :])
        if tp_mlp.fc.linear.bias is not None:
            tp_mlp.fc.linear.bias.copy_(reference_fc.bias[fc_slice])

        in_per_rank = tp_mlp.proj.linear.in_features
        proj_slice = slice(rank * in_per_rank, (rank + 1) * in_per_rank)
        tp_mlp.proj.linear.weight.copy_(reference_proj.weight[:, proj_slice])
        if tp_mlp.proj.bias is not None:
            tp_mlp.proj.bias.copy_(reference_proj.bias)

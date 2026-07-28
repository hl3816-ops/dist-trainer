"""Worker script for the tensor-parallel correctness test.

Run with world_size=1 (plain `python`) to compute and save a full-size
reference MLP's output. Run with world_size=2 (RANK/WORLD_SIZE env vars set,
gloo backend) to compute the same thing via TensorParallelMLP and compare
against the saved reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dist_trainer.tensor_parallel import (
    TensorParallelMLP,
    load_shard_from_reference,
)

N_EMBD = 64
BATCH = 8
SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-out", type=str, required=True)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.manual_seed(SEED)
    x = torch.randn(BATCH, N_EMBD)

    if world_size == 1:
        reference_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        reference_proj = nn.Linear(4 * N_EMBD, N_EMBD)
        with torch.no_grad():
            out = reference_proj(nn.functional.gelu(reference_fc(x)))

        Path(args.reference_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fc_weight": reference_fc.weight,
                "fc_bias": reference_fc.bias,
                "proj_weight": reference_proj.weight,
                "proj_bias": reference_proj.bias,
                "output": out,
            },
            args.reference_out,
        )
        print(
            f"[reference] saved to {args.reference_out}, output shape {tuple(out.shape)}"
        )
        return

    dist.init_process_group(backend="gloo")
    ref = torch.load(args.reference_out, weights_only=True)

    reference_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
    reference_proj = nn.Linear(4 * N_EMBD, N_EMBD)
    with torch.no_grad():
        reference_fc.weight.copy_(ref["fc_weight"])
        reference_fc.bias.copy_(ref["fc_bias"])
        reference_proj.weight.copy_(ref["proj_weight"])
        reference_proj.bias.copy_(ref["proj_bias"])

    tp_mlp = TensorParallelMLP(N_EMBD, world_size)
    load_shard_from_reference(tp_mlp, reference_fc, reference_proj, rank)

    with torch.no_grad():
        tp_out = tp_mlp(x)

    if rank == 0:
        max_diff = (tp_out - ref["output"]).abs().max().item()
        result = {"max_abs_diff": max_diff, "passed": max_diff < 1e-5}
        print(json.dumps(result))
        if not result["passed"]:
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

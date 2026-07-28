"""Throughput/memory benchmark for TinyGPT: measures the actual before/after
effect of mixed precision and gradient checkpointing, and of 1 vs 2 GPUs.

Run single-GPU:
    python examples/benchmark_gpt.py --use-amp --results-out amp_1gpu.json

Run 2-GPU (NCCL):
    torchrun --nproc_per_node=2 examples/benchmark_gpt.py --use-amp --results-out amp_2gpu.json

Reports samples/sec, tokens/sec, and peak CUDA memory for whatever
(use_amp, use_gradient_checkpointing) combination was requested. Run it once
per combination and diff the JSON outputs for the before/after story --
that's intentionally left as a separate step rather than looping over all 4
combos in one process, so a crash/OOM in one config doesn't lose the others.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dist_trainer import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt_data import CharTokenizer, load_text, make_batch_iter
from gpt_model import TinyGPT


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    b, t, v = logits.shape
    return torch.nn.functional.cross_entropy(logits.view(b * t, v), targets.view(b * t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--warmup", type=int, default=10, help="untimed steps before measuring"
    )
    parser.add_argument("--total-batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-grad-checkpoint", action="store_true")
    parser.add_argument("--results-out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA device available, running on CPU (mixed precision will no-op)"
        )
        device = torch.device("cpu")
    else:
        import os

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    text = load_text()
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_gradient_checkpointing=args.use_grad_checkpoint,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device=device, use_amp=args.use_amp)
    backend = "nccl" if device.type == "cuda" else "gloo"
    trainer.setup(backend=backend)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_iter = make_batch_iter(
        data, args.block_size, args.total_batch_size, seed=args.seed
    )

    for _ in range(args.warmup):
        batch = batch_iter(trainer.step, trainer.rank, trainer.world_size)
        trainer.train_step(batch, loss_fn)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if trainer.distributed:
        import torch.distributed as dist

        dist.barrier()

    start = time.perf_counter()
    for _ in range(args.steps):
        batch = batch_iter(trainer.step, trainer.rank, trainer.world_size)
        trainer.train_step(batch, loss_fn)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if trainer.distributed:
        dist.barrier()
    elapsed = time.perf_counter() - start

    samples_per_sec = (args.steps * args.total_batch_size) / elapsed
    tokens_per_sec = samples_per_sec * args.block_size
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    )

    if trainer.rank == 0:
        results = {
            "world_size": trainer.world_size,
            "device": str(device),
            "use_amp": args.use_amp,
            "use_grad_checkpoint": args.use_grad_checkpoint,
            "n_layer": args.n_layer,
            "n_embd": args.n_embd,
            "block_size": args.block_size,
            "total_batch_size": args.total_batch_size,
            "steps_measured": args.steps,
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_cuda_memory_mb": round(peak_mem_mb, 1)
            if peak_mem_mb is not None
            else None,
            "num_params": model.num_params(),
        }
        print(json.dumps(results, indent=2))
        if args.results_out:
            Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results_out).write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

    trainer.teardown()


if __name__ == "__main__":
    main()

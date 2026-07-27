"""CLI entrypoint for the synthetic regression task.

Works unmodified as:
  - single process:  python examples/train.py --steps 200
  - multi process:   torchrun --nproc_per_node=4 examples/train.py --steps 200

because Trainer reads RANK/WORLD_SIZE from the environment (torchrun sets
them; a plain `python` invocation leaves them unset, so it defaults to
world_size=1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dist_trainer import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import RegressionMLP, loss_fn, make_batch_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--total-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--crash-after-step",
        type=int,
        default=None,
        help="raise SystemExit right after this step, to simulate a mid-training crash",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="append per-step {step, loss} JSON lines here",
    )
    parser.add_argument(
        "--save-final-model",
        type=str,
        default=None,
        help="save final model state_dict here (rank 0 only)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = RegressionMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    trainer = Trainer(model, optimizer)
    trainer.setup()

    batch_iter = make_batch_iter(args.total_batch_size, seed=args.seed)

    if not args.no_resume and args.checkpoint_dir:
        trainer.maybe_resume(args.checkpoint_dir)

    log_file = None
    if args.log_path and trainer.rank == 0:
        Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log_path, "a", encoding="utf-8")  # noqa: SIM115 - lifetime matches the training loop below, closed in finally

    try:
        while trainer.step < args.steps:
            batch = batch_iter(trainer.step, trainer.rank, trainer.world_size)
            loss = trainer.train_step(batch, loss_fn)
            if log_file:
                log_file.write(json.dumps({"step": trainer.step, "loss": loss}) + "\n")
                log_file.flush()
            trainer.maybe_checkpoint(args.checkpoint_dir, args.checkpoint_every)

            if (
                args.crash_after_step is not None
                and trainer.step == args.crash_after_step
            ):
                if trainer.rank == 0:
                    print(f"[rank0] simulating crash after step {trainer.step}")
                sys.exit(1)

        if trainer.rank == 0:
            print(f"[rank0] finished at step {trainer.step}")
            if args.save_final_model:
                Path(args.save_final_model).parent.mkdir(parents=True, exist_ok=True)
                underlying = (
                    trainer.model.module
                    if hasattr(trainer.model, "module")
                    else trainer.model
                )
                torch.save(underlying.state_dict(), args.save_final_model)
    finally:
        if log_file:
            log_file.close()
        trainer.teardown()


if __name__ == "__main__":
    main()

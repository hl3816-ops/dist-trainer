"""Does DDP training actually produce the same result as single-process
training, just sharded across processes -- or does it just "run without
crashing" while silently computing something different?

Runs the same task single-process (world_size=1, full batch each step) and
4-process (world_size=4, each rank sees 1/4 of the same per-step batch,
gradients DDP-averaged), then compares final model weights. Comparing losses
wouldn't be a fair test here: each rank's per-shard loss.item() is only over
its own quarter of the batch, so it necessarily differs step-to-step from
the single-process full-batch loss even when the two runs are otherwise
computing the same thing -- the weights are the invariant that should match.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TRAIN_SCRIPT = ROOT / "examples" / "train.py"

STEPS = 50
TOTAL_BATCH_SIZE = 32
SEED = 0
TOLERANCE = 1e-4


def run_single_process(model_out: Path) -> None:
    subprocess.run(
        [
            PY,
            str(TRAIN_SCRIPT),
            "--steps",
            str(STEPS),
            "--total-batch-size",
            str(TOTAL_BATCH_SIZE),
            "--seed",
            str(SEED),
            "--save-final-model",
            str(model_out),
        ],
        check=True,
        cwd=ROOT,
    )


def run_multi_process(model_out: Path, nproc: int, master_port: int = 29511) -> None:
    """Launches `nproc` worker processes directly with subprocess.Popen,
    setting RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT ourselves, rather than
    going through `torchrun`.

    `torchrun`'s own rendezvous bootstrap creates a TCPStore *before* the
    user script even runs, and on this Windows CPU torch build that
    bootstrap unconditionally requests libuv and ignores the USE_LIBUV
    env var (a bug in this specific build, not in this project's code --
    confirmed by triggering the identical failure with a bare `torchrun`
    invocation outside of this test). Launching workers directly and
    setting `env://` init vars ourselves goes through the one call path
    that does honor USE_LIBUV, sidestepping the bug entirely.
    """
    env = {
        **os.environ,
        "USE_LIBUV": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "WORLD_SIZE": str(nproc),
    }
    procs = []
    for rank in range(nproc):
        rank_env = {**env, "RANK": str(rank), "LOCAL_RANK": str(rank)}
        procs.append(
            subprocess.Popen(
                [
                    PY,
                    str(TRAIN_SCRIPT),
                    "--steps",
                    str(STEPS),
                    "--total-batch-size",
                    str(TOTAL_BATCH_SIZE),
                    "--seed",
                    str(SEED),
                    "--save-final-model",
                    str(model_out),
                ],
                cwd=ROOT,
                env=rank_env,
            )
        )
    for rank, p in enumerate(procs):
        returncode = p.wait()
        if returncode != 0:
            raise RuntimeError(f"rank {rank} exited with code {returncode}")


def main():
    tmp = ROOT / ".test_tmp"
    tmp.mkdir(exist_ok=True)
    single_model_path = tmp / "single_final.pt"
    multi_model_path = tmp / "multi_final.pt"

    print(
        f"Running single-process baseline ({STEPS} steps, batch={TOTAL_BATCH_SIZE})..."
    )
    run_single_process(single_model_path)

    print("Running 4-process DDP run (same seed, same effective batch)...")
    run_multi_process(multi_model_path, nproc=4)

    single_state = torch.load(single_model_path, map_location="cpu", weights_only=True)
    multi_state = torch.load(multi_model_path, map_location="cpu", weights_only=True)

    assert single_state.keys() == multi_state.keys(), "parameter names differ"

    max_diff = 0.0
    for name in single_state:
        diff = (single_state[name] - multi_state[name]).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"  {name}: max abs diff = {diff:.2e}")

    print(f"\nmax abs weight difference across all parameters: {max_diff:.2e}")
    assert max_diff < TOLERANCE, (
        f"4-process DDP training diverged from single-process baseline by {max_diff} "
        f"(tolerance {TOLERANCE}) -- gradient sync is not equivalent to single-process training"
    )
    print(
        f"PASS: 4-process DDP training matches single-process baseline within {TOLERANCE}"
    )


if __name__ == "__main__":
    main()

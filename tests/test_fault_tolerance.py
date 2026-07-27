"""Simulates a mid-training crash and confirms resuming from checkpoint
continues training correctly -- not just "doesn't error," but reproduces
the same loss trajectory a run that never crashed would have had.

Two runs, both single-process:
  1. "continuous": trains straight through steps 1..STEPS, logging every step.
  2. "crashed": trains steps 1..CRASH_AT, checkpointing every CHECKPOINT_EVERY
     steps, then the process exits (sys.exit) right after CRASH_AT to
     simulate a killed job. A second invocation of the same script then
     resumes from the last checkpoint and continues to STEPS, appending to
     the same log file.

If checkpoint/resume correctly restores model + optimizer state, the loss
logged at each step number should match between the two runs, because the
per-step training data is deterministic (keyed by step index, see
examples/task.py) and the optimizer picks up exactly where it left off.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TRAIN_SCRIPT = ROOT / "examples" / "train.py"

STEPS = 40
CRASH_AT = 20
CHECKPOINT_EVERY = 5
TOTAL_BATCH_SIZE = 32
SEED = 0
TOLERANCE = 1e-5


def run(extra_args: list[str]) -> None:
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
            *extra_args,
        ],
        check=True,
        cwd=ROOT,
    )


def load_losses(path: Path) -> dict[int, float]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["step"]: row["loss"] for row in rows}


def main():
    tmp = ROOT / ".test_tmp"
    tmp.mkdir(exist_ok=True)
    continuous_log = tmp / "continuous.jsonl"
    crashed_log = tmp / "crashed.jsonl"
    checkpoint_dir = tmp / "crash_ckpt"

    for p in (continuous_log, crashed_log):
        p.unlink(missing_ok=True)
    if checkpoint_dir.exists():
        for f in checkpoint_dir.iterdir():
            f.unlink()

    print(f"Continuous run: {STEPS} steps, no interruption...")
    run(["--log-path", str(continuous_log)])

    print(f"Crashed run: training will exit after step {CRASH_AT} (simulated crash)...")
    result = subprocess.run(
        [
            PY,
            str(TRAIN_SCRIPT),
            "--steps",
            str(STEPS),
            "--total-batch-size",
            str(TOTAL_BATCH_SIZE),
            "--seed",
            str(SEED),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every",
            str(CHECKPOINT_EVERY),
            "--crash-after-step",
            str(CRASH_AT),
            "--log-path",
            str(crashed_log),
        ],
        cwd=ROOT,
        check=False,  # this invocation is expected to exit non-zero (simulated crash)
    )
    assert result.returncode != 0, "expected the crash-simulation run to exit non-zero"
    assert (checkpoint_dir / "latest.pt").exists(), (
        "no checkpoint was written before the crash"
    )

    partial = load_losses(crashed_log)
    print(
        f"  crashed run logged {len(partial)} steps before dying (expected {CRASH_AT})"
    )
    assert max(partial) == CRASH_AT

    print("Resuming from checkpoint and continuing to completion...")
    run(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every",
            str(CHECKPOINT_EVERY),
            "--log-path",
            str(crashed_log),
        ]
    )

    continuous = load_losses(continuous_log)
    resumed = load_losses(crashed_log)

    assert set(continuous.keys()) == set(resumed.keys()) == set(range(1, STEPS + 1)), (
        "step coverage differs between the continuous and crashed+resumed runs"
    )

    max_diff = max(abs(continuous[s] - resumed[s]) for s in continuous)
    print(f"max abs loss difference at any step: {max_diff:.2e}")
    assert max_diff < TOLERANCE, (
        f"crash+resume run diverged from the continuous baseline by {max_diff} "
        f"(tolerance {TOLERANCE}) -- checkpoint/resume is not faithfully restoring state"
    )
    print(f"PASS: crash+resume run matches the continuous baseline within {TOLERANCE}")
    print(
        f"  (lost zero training progress across a simulated crash at step {CRASH_AT}/{STEPS})"
    )


if __name__ == "__main__":
    main()

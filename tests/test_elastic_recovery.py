"""Does the training group actually recover automatically from a worker
crash -- detect it, kill the survivors, restart the whole group, and pick
back up from the last checkpoint -- or does it just hang / need a human to
notice and re-run the command?

2-rank CPU group training examples/task.py's regression MLP. On the first
supervised attempt, rank 1 is configured to crash partway through; nothing
external re-invokes anything -- dist_trainer.supervisor.run_with_auto_restart
must detect the failure, kill rank 0, and relaunch both ranks itself. The
restarted group resumes from the last shared checkpoint.

The comparison baseline is a *continuous 2-rank run*, not single-process:
rank 0's logged loss is only ever computed over its own half of the batch
(see tests/test_correctness.py's docstring), so it would never match a
single-process full-batch loss even when training is otherwise identical.
Comparing 2-rank vs. 2-rank keeps that basis consistent -- the same shape
of check tests/test_fault_tolerance.py does for the single-process case,
extended to the auto-restarted multi-process case.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dist_trainer.supervisor import run_with_auto_restart

PY = sys.executable
TRAIN_SCRIPT = ROOT / "examples" / "train.py"

STEPS = 40
CRASH_AT = 15
CHECKPOINT_EVERY = 5
TOTAL_BATCH_SIZE = 32
SEED = 0
TOLERANCE = 1e-4


def _launch_cmd_factory(checkpoint_dir: Path, log_path: Path, inject_crash: bool):
    def launch_cmd(rank: int, attempt: int) -> list[str]:
        args = [
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
        ]
        if rank == 0:
            args += ["--log-path", str(log_path)]
        if inject_crash and attempt == 0 and rank == 1:
            args += ["--crash-after-step", str(CRASH_AT)]
        return args

    def env_fn(rank: int, attempt: int) -> dict:
        return {
            "RANK": str(rank),
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29514",
        }

    return launch_cmd, env_fn


def _read_log(path: Path) -> dict[int, float]:
    with open(path, encoding="utf-8") as f:
        return {
            json.loads(line)["step"]: json.loads(line)["loss"]
            for line in f
            if line.strip()
        }


def run_continuous_2rank(log_path: Path, checkpoint_dir: Path) -> dict[int, float]:
    launch_cmd, env_fn = _launch_cmd_factory(
        checkpoint_dir, log_path, inject_crash=False
    )
    procs = [
        subprocess.Popen(
            launch_cmd(rank, attempt=0), env={**os.environ, **env_fn(rank, 0)}
        )
        for rank in range(2)
    ]
    for rank, p in enumerate(procs):
        if p.wait() != 0:
            raise RuntimeError(f"baseline rank {rank} failed")
    return _read_log(log_path)


def main():
    tmp = ROOT / ".test_tmp"
    tmp.mkdir(exist_ok=True)
    recovered_ckpt = tmp / "elastic_ckpt_recovered"
    baseline_ckpt = tmp / "elastic_ckpt_baseline"
    recovered_log = tmp / "elastic_recovered.jsonl"
    baseline_log = tmp / "elastic_baseline.jsonl"

    for p in (recovered_log, baseline_log):
        p.unlink(missing_ok=True)
    for d in (recovered_ckpt, baseline_ckpt):
        if d.exists():
            for f in d.iterdir():
                f.unlink()

    print(
        f"Continuous 2-rank baseline: {STEPS} steps, batch={TOTAL_BATCH_SIZE}, no crash..."
    )
    baseline = run_continuous_2rank(baseline_log, baseline_ckpt)

    print(
        f"Supervised 2-rank group: rank 1 will crash at step {CRASH_AT} on attempt 0, "
        "recovery must be automatic..."
    )
    launch_cmd, env_fn = _launch_cmd_factory(
        recovered_ckpt, recovered_log, inject_crash=True
    )
    history = run_with_auto_restart(
        launch_cmd, world_size=2, max_restarts=2, env_fn=env_fn
    )

    print(f"attempts: {[(a.attempt, a.returncodes, a.succeeded) for a in history]}")
    assert len(history) == 2, (
        f"expected exactly 1 crash + 1 successful retry, got {history}"
    )
    assert not history[0].succeeded, (
        "expected the first attempt to fail (simulated crash)"
    )
    assert history[1].succeeded, (
        "expected the second (auto-restarted) attempt to succeed"
    )
    assert recovered_ckpt.joinpath("latest.pt").exists(), (
        "no checkpoint survived the crash"
    )

    recovered = _read_log(recovered_log)

    assert set(recovered.keys()) == set(baseline.keys()) == set(range(1, STEPS + 1)), (
        "step coverage differs between the baseline and the auto-recovered run"
    )
    max_diff = max(abs(baseline[s] - recovered[s]) for s in baseline)
    print(f"max abs loss difference vs. continuous 2-rank baseline: {max_diff:.2e}")
    assert max_diff < TOLERANCE, (
        f"auto-recovered training diverged from the baseline by {max_diff} "
        f"(tolerance {TOLERANCE})"
    )
    print(
        f"PASS: crash at step {CRASH_AT}/{STEPS} was auto-detected, group auto-restarted, "
        f"resumed from checkpoint, and matched the uninterrupted 2-rank baseline within {TOLERANCE}"
    )


if __name__ == "__main__":
    main()

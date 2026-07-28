"""Does the tensor-parallel MLP (dist_trainer/tensor_parallel.py) compute
the exact same thing as a full-size reference MLP, split across ranks and
recombined with exactly one all-reduce -- or does it just "run"?

Single process computes a reference MLP's output. Two processes (CPU/gloo,
launched directly via subprocess.Popen for the same reason as
test_correctness.py -- see that file's docstring for the torchrun/libuv
issue this sidesteps) load the same weights sharded across ranks via
ColumnParallelLinear -> RowParallelLinear, and rank 0 compares its final
output to the reference.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
WORKER_SCRIPT = ROOT / "examples" / "tp_worker.py"


def main():
    tmp = ROOT / ".test_tmp"
    tmp.mkdir(exist_ok=True)
    reference_path = tmp / "tp_reference.pt"
    reference_path.unlink(missing_ok=True)

    print("Computing single-process reference MLP output...")
    subprocess.run(
        [PY, str(WORKER_SCRIPT), "--reference-out", str(reference_path)],
        check=True,
        cwd=ROOT,
    )

    print("Running 2-rank tensor-parallel MLP (CPU/gloo)...")
    world_size = 2
    env_base = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29513",
        "WORLD_SIZE": str(world_size),
    }
    procs = []
    for rank in range(world_size):
        rank_env = {**env_base, "RANK": str(rank)}
        procs.append(
            subprocess.Popen(
                [PY, str(WORKER_SCRIPT), "--reference-out", str(reference_path)],
                cwd=ROOT,
                env=rank_env,
                stdout=subprocess.PIPE,
                text=True,
            )
        )

    rank0_stdout = None
    for rank, p in enumerate(procs):
        stdout, _ = p.communicate()
        if rank == 0:
            rank0_stdout = stdout
        if p.returncode != 0:
            raise RuntimeError(
                f"rank {rank} exited with code {p.returncode}:\n{stdout}"
            )

    result_line = next(
        line for line in rank0_stdout.splitlines() if line.strip().startswith("{")
    )
    result = json.loads(result_line)
    print(f"max abs diff vs single-process reference: {result['max_abs_diff']:.2e}")
    assert result["passed"], (
        f"tensor-parallel MLP diverged from the reference by {result['max_abs_diff']}"
    )
    print("PASS: 2-rank tensor-parallel MLP matches the single-process reference")


if __name__ == "__main__":
    main()

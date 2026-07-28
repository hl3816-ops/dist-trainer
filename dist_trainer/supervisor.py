"""Automatic fault recovery: detect a crashed worker and restart the whole
process group, rather than requiring a human (or a script invoked by one)
to notice and re-run the command.

`tests/test_fault_tolerance.py` proves checkpoint/resume itself is correct
-- that resuming *does* pick up where training left off. This module is the
other half: actually noticing a worker died and relaunching automatically,
which is what "robust under rapid iteration" means in production rather
than in a single-process manual demo. Combined with checkpointing, a
restarted group resumes from the last checkpoint rather than from scratch.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RestartAttempt:
    attempt: int
    returncodes: list[int]
    succeeded: bool


def run_with_auto_restart(
    launch_cmd: Callable[[int, int], list[str]],
    world_size: int,
    max_restarts: int = 3,
    poll_interval: float = 0.2,
    env_fn: Callable[[int, int], dict] | None = None,
) -> list[RestartAttempt]:
    """Launches `world_size` processes via `launch_cmd(rank, attempt)`
    (each process's environment is the current process's environment
    merged with `env_fn(rank, attempt)`, if given), monitors them, and if
    ANY exits non-zero, kills the rest of that generation and relaunches
    the whole group -- up to `max_restarts` additional attempts. Returns
    the per-attempt history (for tests/observability); raises if the group
    never succeeds.

    This deliberately restarts the *whole* group rather than just the dead
    rank: NCCL/gloo process groups don't support hot-replacing a single
    member mid-training, so a clean restart of everyone (picking back up
    from the last shared checkpoint) is the actual production pattern, not
    a simplification.
    """
    history: list[RestartAttempt] = []

    for attempt in range(max_restarts + 1):
        procs = []
        for rank in range(world_size):
            env = {**os.environ, **env_fn(rank, attempt)} if env_fn else None
            procs.append(subprocess.Popen(launch_cmd(rank, attempt), env=env))

        returncodes: list[int | None] = [None] * world_size
        while any(rc is None for rc in returncodes):
            for i, p in enumerate(procs):
                if returncodes[i] is None and p.poll() is not None:
                    returncodes[i] = p.returncode
                    if p.returncode != 0:
                        # one worker died -- kill the rest of this
                        # generation immediately rather than letting them
                        # run on with a now-broken process group
                        for other in procs:
                            if other.poll() is None:
                                other.kill()
            if any(rc is None for rc in returncodes):
                time.sleep(poll_interval)

        succeeded = all(rc == 0 for rc in returncodes)
        history.append(
            RestartAttempt(
                attempt=attempt, returncodes=list(returncodes), succeeded=succeeded
            )
        )
        if succeeded:
            return history

    raise RuntimeError(
        f"training did not succeed after {max_restarts + 1} attempts: "
        f"{[a.returncodes for a in history]}"
    )

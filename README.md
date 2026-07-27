# dist-trainer

A small, reusable distributed-training library, built to demonstrate (at CPU/
laptop scale, with real `torch.distributed` mechanics) the core capabilities
a training-infrastructure role actually needs day to day: correct
distributed gradient sync, a reusable framework rather than one-off scripts,
and fault tolerance under process failure.

## Why this exists

Four capabilities, four pieces of concrete evidence:

| Capability | Where it's proven |
|---|---|
| Distributed training that's actually correct, not just "runs" | [`tests/test_correctness.py`](tests/test_correctness.py) -- 4-process DDP vs. single-process baseline, final weights match to **5.96e-08** |
| Reusable framework, not a one-off script | [`dist_trainer/trainer.py`](dist_trainer/trainer.py) -- domain-agnostic `Trainer`; the example task in `examples/` is ~60 lines that just plugs into it |
| Reliability under failure ("robust under rapid iteration") | [`tests/test_fault_tolerance.py`](tests/test_fault_tolerance.py) -- kill the process mid-training, resume from checkpoint, **zero** loss trajectory divergence (0.00e+00) vs. an uninterrupted run |
| Debugging real distributed-systems infrastructure, not just model code | Found and worked around a real bug in `torchrun`'s Windows rendezvous bootstrap (see "A real bug found along the way" below) |

## Architecture

```
dist_trainer/            <- the reusable library (domain-agnostic)
  trainer.py                Trainer: DDP setup/teardown, train_step, checkpoint hooks
  checkpoint.py              atomic checkpoint save/load (model + optimizer + RNG state)

examples/                <- a concrete task plugged into the library
  task.py                    tiny synthetic regression MLP + deterministic data
  train.py                   CLI entrypoint; identical code path whether run
                              single-process or multi-process

tests/
  test_correctness.py        DDP vs. single-process weight equivalence check
  test_fault_tolerance.py    crash + resume vs. continuous-run equivalence check
```

The `Trainer` reads `RANK` / `WORLD_SIZE` from the environment and only
wraps the model in DDP when `WORLD_SIZE > 1` -- the exact same `train.py`
runs unmodified single-process (`python examples/train.py`) or distributed
(`torchrun --nproc_per_node=N examples/train.py`).

## Quickstart

```bash
pip install -r requirements.txt

# single process
python examples/train.py --steps 200

# multi-process (4 workers on this machine)
torchrun --nproc_per_node=4 examples/train.py --steps 200

# with checkpointing (survives being killed and restarted)
python examples/train.py --steps 200 --checkpoint-dir ckpts --checkpoint-every 20

# run the two correctness proofs
python tests/test_correctness.py
python tests/test_fault_tolerance.py
```

## A real bug found along the way

`torchrun`'s built-in rendezvous bootstrap creates its own `TCPStore`
*before* the user's script even runs, to coordinate workers. On the Windows
CPU build of torch used here, that bootstrap unconditionally requests the
libuv backend and ignores the documented `USE_LIBUV=0` escape hatch --
confirmed by reproducing the identical failure with a bare `torchrun`
invocation, independent of anything in this project. `tests/test_correctness.py`
works around it by launching worker processes directly with `subprocess.Popen`
and setting `RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` env vars
itself, which routes through the one `TCPStore` construction path in
`torch.distributed` that *does* honor `USE_LIBUV` -- sidestepping the bug
rather than fixing torch itself. (`torchrun` works normally on Linux/GPU
hosts, e.g. Kaggle -- this is specific to this Windows CPU wheel.)

## Design choices / limitations

- **CPU-only, `gloo` backend, single machine.** This proves the distributed
  *mechanics* (process groups, DDP gradient averaging, multi-process
  coordination, checkpoint/resume) are correctly implemented -- it does not
  demonstrate GPU throughput/memory optimization (mixed precision, NCCL,
  multi-node scaling), which needs actual GPU hardware. A natural follow-up
  is running the same `Trainer` on a multi-GPU host (e.g. Kaggle's free
  2x T4) with `backend="nccl"` and profiling throughput before/after
  optimizations like mixed precision and gradient checkpointing.
- **Correctness check compares final weights, not losses**, because each
  rank's `loss.item()` is only over its own shard of the batch and will
  legitimately differ step-to-step from a single-process full-batch loss
  even when training is otherwise equivalent -- the post-training weights
  are the actual invariant that should match.
- **Checkpointing assumes shared storage** (all ranks read the same path),
  true for a single machine or multi-node with a shared filesystem (NFS,
  etc.) -- the common real-world setup, but worth stating explicitly.

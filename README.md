# dist-trainer

A small, reusable distributed-training library, built to demonstrate the
core capabilities a training-infrastructure role actually needs day to day:
correct distributed gradient sync, a reusable framework rather than one-off
scripts, fault tolerance under process failure, and real GPU performance
optimization -- backed by measurements on real 2x T4 GPUs, not just claims.

## Why this exists

Four capabilities, four pieces of concrete evidence:

| Capability | Where it's proven |
|---|---|
| Distributed training that's actually correct, not just "runs" | [`tests/test_correctness.py`](tests/test_correctness.py) -- 4-process DDP vs. single-process baseline, final weights match to **5.96e-08** |
| Reusable framework, not a one-off script | [`dist_trainer/trainer.py`](dist_trainer/trainer.py) -- domain-agnostic `Trainer` (DDP, AMP, checkpointing); both the toy-MLP and the transformer example plug into the same ~150-line class |
| Reliability under failure ("robust under rapid iteration") | [`tests/test_fault_tolerance.py`](tests/test_fault_tolerance.py) -- kill the process mid-training, resume from checkpoint, **zero** loss trajectory divergence (0.00e+00) vs. an uninterrupted run |
| High-performance optimization (real GPU numbers) | [`kaggle_kernel/benchmark_kernel.py`](kaggle_kernel/benchmark_kernel.py), run on real 2x T4 GPUs -- mixed precision: **2.6x** throughput; gradient checkpointing: **68% less** peak memory; 2 GPUs vs 1: **1.79x** throughput. See "Real GPU benchmark results" below. |

## Architecture

```
dist_trainer/            <- the reusable library (domain-agnostic)
  trainer.py                Trainer: DDP setup/teardown, train_step (+AMP), checkpoint hooks
  checkpoint.py              atomic checkpoint save/load (model + optimizer + RNG state)

examples/
  task.py, train.py          toy regression MLP -- used by the correctness/fault-tolerance tests
  gpt_model.py                minimal decoder-only transformer (causal self-attention,
                              optional gradient checkpointing per block)
  gpt_data.py                  character-level tiny-shakespeare data loading
  benchmark_gpt.py            throughput/memory benchmark CLI (single or multi-GPU via torchrun)

kaggle_kernel/            <- self-contained script + metadata to run the same
  benchmark_kernel.py        benchmark on real Kaggle GPUs (inlines model/data/
  kernel-metadata.json       training code so the push is a single self-sufficient file)

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

# throughput/memory benchmark (needs a real CUDA GPU; runs but is meaningless on CPU)
python examples/benchmark_gpt.py --use-amp --use-grad-checkpoint --results-out bench.json
torchrun --nproc_per_node=2 examples/benchmark_gpt.py --use-amp --results-out bench_2gpu.json
```

## Real GPU benchmark results

Run on a Kaggle notebook with 2x NVIDIA T4 GPUs (`kaggle_kernel/benchmark_kernel.py`,
6-layer/384-dim TinyGPT, batch size 64, block size 128, 40 measured steps after
10 warmup steps):

| GPUs | Mixed precision | Grad checkpoint | Samples/sec | Tokens/sec | Peak memory (MB) |
|-----:|:---:|:---:|-----:|-----:|-----:|
| 1 | | | 368.0 | 47,110 | 1,597.6 |
| 1 | | Y | 289.4 | 37,048 | **506.3** |
| 1 | Y | | **952.2** | **121,884** | 1,150.3 |
| 1 | Y | Y | 751.8 | 96,230 | 414.7 |
| 2 | | | 658.6 | 84,298 | 963.7 |
| 2 | | Y | 528.8 | 67,681 | 417.7 |
| 2 | Y | | 1,475.3 | 188,839 | 714.0 |
| 2 | Y | Y | 1,189.1 | 152,202 | **341.1** |

Reading the table:
- **Mixed precision (1 GPU): 2.6x throughput** (368 -> 952 samples/sec) *and* lower
  peak memory (1,598MB -> 1,150MB) -- fp16 activations are smaller, and T4's tensor
  cores are built for fp16 matmuls, so this is close to a pure win here.
- **Gradient checkpointing (1 GPU, no AMP): 68% less peak memory** (1,598MB -> 506MB)
  for a 21% throughput cost (recomputing activations during backward instead of
  storing them) -- the classic compute-for-memory trade, useful when memory (not
  compute) is the constraint on a bigger model.
- **2 GPUs vs. 1 (no AMP/checkpoint): 1.79x throughput**, not a clean 2x --
  gradient all-reduce communication overhead, as expected, not a bug.
- **Stacking everything (2 GPU + AMP + grad checkpoint): 341.1MB peak memory**
  (79% less than the 1-GPU unoptimized baseline) while still running **3.2x**
  faster than that baseline -- the useful combination when memory is the binding
  constraint and you still want throughput back via more GPUs.

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

Two more platform-specific gotchas hit while getting the GPU benchmark
running on Kaggle, in case they save someone else the debugging time:
- `torch.cuda.is_available()` was `False` and every CUDA call failed on the
  first two kernel runs despite the account showing 30 GPU-hours of quota --
  Kaggle gates *using* GPU sessions behind phone verification separately
  from just having quota allocated to the account.
- `kaggle kernels push` on an update (not first creation) 409s if the
  `kernel-metadata.json` `id` field's slug doesn't exactly match the slug
  Kaggle actually derived from the kernel's title on creation -- the fix is
  making `id` match the real slug from the kernel's URL, not the slug you
  intended.

## Design choices / limitations

- **Correctness check compares final weights, not losses**, because each
  rank's `loss.item()` is only over its own shard of the batch and will
  legitimately differ step-to-step from a single-process full-batch loss
  even when training is otherwise equivalent -- the post-training weights
  are the actual invariant that should match.
- **Checkpointing assumes shared storage** (all ranks read the same path),
  true for a single machine or multi-node with a shared filesystem (NFS,
  etc.) -- the common real-world setup, but worth stating explicitly.
- **Still single-node (2 GPUs, 1 machine) and a small model (~11M params).**
  This demonstrates the mechanics and the direction/magnitude of each
  optimization correctly, but the specific numbers won't generalize to
  multi-node training or to models where activation memory, not the model
  itself, dominates -- that's where sharding techniques like FSDP/ZeRO
  (not implemented here) start to matter more than DDP + gradient
  checkpointing.

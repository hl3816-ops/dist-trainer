# dist-trainer

A small, reusable distributed-training library, built to demonstrate the
core capabilities a training-infrastructure role actually needs day to day:
correct distributed gradient sync (data-parallel *and* tensor-parallel),
a reusable framework rather than one-off scripts, fault tolerance that
recovers on its own rather than needing a human to notice, and real GPU
performance optimization -- backed by measurements on real 2x T4 GPUs, not
just claims.

## Why this exists

| Capability | Where it's proven |
|---|---|
| Distributed training that's actually correct, not just "runs" | [`tests/test_correctness.py`](tests/test_correctness.py) -- 4-process DDP vs. single-process baseline (CPU/gloo), final weights match to **5.96e-08**; re-verified on real 2x T4 GPUs (NCCL) at **1.19e-07** |
| ...and not just data parallelism | [`tests/test_tensor_parallel_correctness.py`](tests/test_tensor_parallel_correctness.py) -- a Megatron-LM-style tensor-parallel MLP (splits individual layers, not just data, across ranks) matches a full-size reference to **1.79e-07** |
| ...and not just one framework | [`tests/test_jax_correctness.py`](tests/test_jax_correctness.py) -- the same data-parallel correctness invariant, in JAX via `jax.pmap`/`jax.lax.pmean` instead of PyTorch DDP, matches to **5.96e-08** |
| Reusable framework, not a one-off script | [`dist_trainer/trainer.py`](dist_trainer/trainer.py) -- domain-agnostic `Trainer` (DDP or FSDP, AMP, checkpointing); both the toy-MLP and the transformer example plug into the same ~170-line class |
| Reliability that's *automatic*, not just correct | [`tests/test_fault_tolerance.py`](tests/test_fault_tolerance.py) proves checkpoint/resume itself is exact (0.00e+00 divergence); [`tests/test_elastic_recovery.py`](tests/test_elastic_recovery.py) proves the *detection and restart* is automatic too -- a crashed worker is noticed and the whole group is relaunched with zero external intervention, matching an uninterrupted baseline exactly |
| High-performance optimization (real GPU numbers) | [`kaggle_kernel/benchmark_kernel.py`](kaggle_kernel/benchmark_kernel.py), run on real 2x T4 GPUs -- mixed precision: **2.6x** throughput; gradient checkpointing: **68% less** peak memory; 2 GPUs vs 1: **1.79x** throughput |

## Architecture

```
dist_trainer/            <- the reusable PyTorch library (domain-agnostic)
  trainer.py                Trainer: DDP/FSDP setup+teardown, train_step (+AMP), checkpoint hooks
  checkpoint.py              atomic checkpoint save/load (model + optimizer + RNG state)
  tensor_parallel.py         ColumnParallelLinear / RowParallelLinear / TensorParallelMLP
  supervisor.py               run_with_auto_restart: detect a crashed worker, restart the group

examples/
  task.py, train.py          toy regression MLP -- used by the correctness/fault-tolerance/
                              elastic-recovery tests
  tp_worker.py                worker script for the tensor-parallel correctness test
  gpt_model.py                minimal decoder-only transformer (causal self-attention,
                              optional gradient checkpointing per block)
  gpt_data.py                  character-level tiny-shakespeare data loading
  benchmark_gpt.py            throughput/memory benchmark CLI (single or multi-GPU via torchrun)

examples_jax/             <- cross-framework counterpart: same data-parallel correctness
  model.py, train.py         invariant, in JAX (jax.pmap + jax.lax.pmean) instead of PyTorch

kaggle_kernel/            <- self-contained scripts + metadata to run on real Kaggle GPUs
  benchmark_kernel.py         AMP / grad-checkpoint throughput+memory matrix (1 & 2 GPU)
  gpu_extras/
    gpu_extras_kernel.py       DDP-vs-single-process correctness on NCCL, and FSDP vs DDP
  profiling/
    profile_kernel.py          torch.profiler trace: which NCCL kernels DDP/FSDP actually call
  (each inlines its own model/data/training code so the push is one self-sufficient file)

tests/
  test_correctness.py                    DDP vs. single-process weight equivalence
  test_tensor_parallel_correctness.py    tensor-parallel MLP vs. a full-size reference
  test_jax_correctness.py                 jax.pmap vs. single-device JAX training
  test_fault_tolerance.py                crash + resume vs. continuous-run equivalence
  test_elastic_recovery.py                automatic crash detection + group restart
```

The `Trainer` reads `RANK` / `WORLD_SIZE` from the environment and only
wraps the model when `WORLD_SIZE > 1` -- the exact same `train.py`
runs unmodified single-process (`python examples/train.py`) or distributed
(`torchrun --nproc_per_node=N examples/train.py`). Pass
`trainer.setup(parallelism="fsdp")` instead of the default `"ddp"` to shard
parameters/gradients/optimizer state across ranks instead of replicating
the full model on each one (checkpointing an FSDP-wrapped `Trainer` isn't
supported yet -- see limitations).

## Quickstart

```bash
pip install -r requirements.txt

# single process
python examples/train.py --steps 200

# multi-process (4 workers on this machine)
torchrun --nproc_per_node=4 examples/train.py --steps 200

# with checkpointing (survives being killed and restarted)
python examples/train.py --steps 200 --checkpoint-dir ckpts --checkpoint-every 20

# run the correctness / fault-tolerance proofs (all CPU-only, no GPU needed)
python tests/test_correctness.py
python tests/test_tensor_parallel_correctness.py
python tests/test_fault_tolerance.py
python tests/test_elastic_recovery.py

# JAX cross-framework correctness proof (separate deps: pip install -r examples_jax/requirements.txt)
python tests/test_jax_correctness.py

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

## GPU correctness, and FSDP vs. DDP

`kaggle_kernel/gpu_extras/gpu_extras_kernel.py`, same 2x T4 setup:

**DDP correctness on NCCL** (not just CPU/gloo): 2-process DDP vs. single-process,
30 steps, same TinyGPT config -- final weights match to **1.19e-07** max abs
difference. Same invariant as `tests/test_correctness.py`, now verified on the
actual backend (NCCL) and hardware (GPU) real training uses, not just the
portable CPU stand-in.

**FSDP vs. DDP, at two model sizes** (2 GPUs, AdamW, no AMP/checkpointing, 30 steps)
-- run at both the ~11M-param config used everywhere else in this repo, and a
~303M-param config (24 layers, 1024 dim -- as large as comfortably fits DDP's
full-replica memory on a single 16GB T4), to see the *trend*, not just one
data point:

| Params | Parallelism | Samples/sec | Peak memory (MB) |
|---:|---|---:|---:|
| 10.7M | DDP | 675.2 | 919.0 |
| 10.7M | FSDP | 641.1 | **852.5** (-7.2%) |
| 302.6M | DDP | 27.9 | 8,561.7 |
| 302.6M | FSDP | 21.1 | **6,748.1** (-21.2%) |

Scaling the model ~28x turned FSDP's memory advantage from a modest 7.2% into
a real 21.2% -- confirming the mechanism (FSDP shards optimizer state instead
of replicating it, and optimizer state is a bigger fraction of memory as the
model grows). But the throughput cost grew too, from 5.1% to 24.4%: FSDP has
to all-gather the full parameters back together for every forward/backward,
and that communication cost scales with model size just like the memory
savings do. This is the honest trade, not a free lunch -- FSDP wins on models
too big for DDP to fit at all, not universally. (~303M params is still far
below what "large model" means in the industry -- see limitations below.)

## Tensor parallelism: splitting layers, not just data

DDP and FSDP are both *data*-parallel: every rank runs the same full layer,
just on different data (DDP) or with different shards of that layer's state
at rest (FSDP). Neither helps if a single layer's activations don't fit on
one GPU at all -- that's what tensor parallelism is for, and it's a
different technique, not a bigger version of the same one.

[`dist_trainer/tensor_parallel.py`](dist_trainer/tensor_parallel.py) implements
the standard Megatron-LM MLP pattern: `ColumnParallelLinear` splits a layer's
*output* features across ranks (no communication needed -- each rank just
computes its own disjoint slice), feeding directly into `RowParallelLinear`,
which splits its *input* features to match and all-reduces the partial sums
back together. The column/row split is chosen specifically so the
intermediate activation between the two layers never needs to be gathered --
exactly one all-reduce per MLP block, regardless of how many ranks.

`tests/test_tensor_parallel_correctness.py` proves it against a full-size
reference MLP with the same weights sharded across 2 ranks: max abs output
difference **1.79e-07** (CPU/gloo, same subprocess-launch pattern as
`test_correctness.py`, no GPU needed for the correctness proof itself).

## Automatic fault recovery: noticing failure, not just surviving it

`test_fault_tolerance.py` (above) proves checkpoint/resume is *exact* --
resuming really does pick up where training left off. It doesn't prove
anything notices a crash in the first place; that test's "resume" is
triggered by a human (or a script standing in for one) re-invoking the
command.

[`dist_trainer/supervisor.py`](dist_trainer/supervisor.py)'s
`run_with_auto_restart` closes that gap: it launches a process group,
monitors it, and if *any* rank exits non-zero, kills the rest of that
generation and relaunches the whole group itself -- up to a configurable
number of attempts. (Restarting the *whole* group rather than hot-replacing
the one dead rank is deliberate: NCCL/gloo process groups don't support
that, so a clean full restart picking back up from the last shared
checkpoint is the actual production pattern, not a simplification.)

`tests/test_elastic_recovery.py` proves the combination end to end: a
2-rank group where rank 1 is configured to crash partway through is
auto-detected, the whole group is killed and relaunched with **zero**
external intervention, and the recovered run's loss trajectory matches a
continuous (never-crashed) 2-rank baseline **exactly** (0.00e+00 diff).

## Cross-framework: the same invariant, in JAX

The JD this project targets lists "PyTorch, JAX" together, so
[`examples_jax/`](examples_jax/) proves the same core claim --
distributed data-parallel training reproduces single-device training --
using JAX's actual idioms instead of a PyTorch-shaped imitation: explicit
parameter pytrees (no stateful `nn.Module`), `jax.pmap` to compile the
training step once and run it in parallel across devices, and
`jax.lax.pmean` inside the pmapped function as JAX's equivalent of DDP's
gradient all-reduce. Devices are simulated on CPU via
`XLA_FLAGS=--xla_force_host_platform_device_count=N` (set in
[`examples_jax/train.py`](examples_jax/train.py)), so -- like the tensor-parallel and
elastic-recovery proofs -- this needs no GPU to demonstrate the real
multi-device mechanics.

`tests/test_jax_correctness.py`: 4-simulated-device `pmap` training vs.
single-device, max abs parameter difference **5.96e-08**.

## What's actually inside that communication cost: a profiler trace

The numbers above say *how much* FSDP's communication overhead costs; they
don't say *what it's actually doing*. `kaggle_kernel/profiling/profile_kernel.py`
wraps a few training steps of each strategy (303M-param config, 2 GPUs) in
`torch.profiler` and reports the real CUDA kernels, not an inference from
throughput deltas:

| Parallelism | Communication ops | % of CUDA time |
|---|---|---:|
| DDP | `ncclDevKernel_AllReduce` only | 23.2% |
| FSDP | `ncclDevKernel_AllGather` + `ncclDevKernel_ReduceScatter` | 18.5% |

This is the mechanistic confirmation of why the two strategies have different
costs: DDP does exactly one kind of communication per step (all-reduce the
full gradient). FSDP does two -- all-gather the full parameters back together
before forward/backward (since it only stores a shard at rest), then reduce-
scatter the gradients back into shards afterward. Two collective calls
instead of one is the concrete mechanism behind FSDP's extra overhead, not
just an abstract "sharding has a cost."

**Caveat that matters**: the profiler's own instrumentation isn't free --
per-step time under `torch.profiler` here was ~4x slower than the same config
measured cleanly in the un-instrumented benchmark above (profiling records
every op and its call stack, which is itself GPU/CPU work). So the *absolute*
times in this table shouldn't be compared against the throughput numbers
elsewhere in this README -- only the *relative* communication-vs-compute
split and the *identity* of which kernels are running are meaningful here.
Conflating "what a profiler measures" with "what actually happens
unobserved" is exactly the kind of mistake that produces wrong conclusions
from real profiling data, so it's worth stating explicitly rather than
leaving it implicit.

## Scaling beyond one node

Everything above was measured on a single machine. [`MULTI_NODE_DESIGN.md`](MULTI_NODE_DESIGN.md)
is a technical design document covering what would actually have to
change: rendezvous/launcher config across real hosts, NCCL transport
selection and why intra- vs. inter-node bandwidth asymmetry is the real
scaling bottleneck, gradient bucketing/overlap tuning, fault recovery when
node failure stops being an edge case, and sharded/async checkpointing and
streaming data loading at a scale where this repo's simplifications
(single-file synchronous checkpoints, one small in-memory dataset) stop
being fine.

**A real 2-node attempt, and why it's still blocked on infra rather than
code.** I rented two separate RunPod GPU instances (not two processes on
one machine) specifically to validate `dist.init_process_group` across a
real network boundary, and hit -- then precisely root-caused -- a platform
networking limitation rather than getting a clean pass:

- Cloud GPU providers like RunPod expose only the SSH port between
  separate pods; there's no shared private network reachable between two
  instances by default.
- An SSH local/remote port-forward tunnel (`-L`/`-R`) successfully bridged
  the rendezvous handshake (`TCPStore` on the fixed `MASTER_PORT`) -- the
  two ranks *could* find each other.
- But `gloo`'s actual data-channel step (`connectFullMesh`) doesn't reuse
  that address: each rank auto-detects and advertises its own
  container-internal IP (e.g. `172.21.0.2`), and the other rank tries to
  connect directly to it on a dynamically chosen port. That address only
  exists inside the originating pod's private Docker network, so the
  connection times out -- confirmed by reading the exact IP:port gloo
  logged in the `connectFullMesh` failure.
- The general fix for arbitrary-port cross-host routing is a transparent
  VPN-style tunnel (tried `sshuttle`), but that requires `CAP_NET_ADMIN`
  inside the container to install NAT/firewall rules, which RunPod's
  containers explicitly strip (confirmed via `/proc/self/status`'s
  capability bounding set) -- not something fixable from inside the
  container at any privilege level available to a renter.

So the rendezvous half of real multi-node coordination is validated
end-to-end; the full-mesh data-plane half is blocked by the specific cloud
provider's container networking, not by anything in this library. The
fix would be a provider with real inter-node private networking (e.g. a
proper VPC/cluster product), not more debugging inside these containers.

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
- **Still single-node (2 GPUs, 1 machine), and "big" here means ~303M params**,
  the most that comfortably fits DDP's full-replica memory on a single 16GB
  T4 -- nowhere close to an industry "large model" (billions of params,
  many nodes). The 11M -> 303M comparison shows the *direction* of the FSDP
  memory/throughput trade correctly (see "GPU correctness, and FSDP vs. DDP"
  above), but the specific percentages will keep shifting at real large-model
  scale, and multi-node effects (network topology, cross-node all-gather
  latency) aren't represented here at all.
- **FSDP checkpointing isn't implemented.** `dist_trainer/checkpoint.py`
  assumes a plain or DDP-wrapped model's `state_dict()`; FSDP's is sharded
  by default and needs an explicit `state_dict_type` context to gather a
  full checkpoint, which this library doesn't do yet.
- **Tensor parallelism covers one MLP block**, not a full transformer (the
  attention layer, and combining tensor parallelism with DDP/FSDP the way
  real 3D-parallel training does, aren't implemented) -- enough to prove
  the column/row-split mechanism is correctly implemented and verified,
  not a drop-in replacement for `examples/gpt_model.py`'s attention blocks.
- **Elastic recovery is single-node.** `dist_trainer/supervisor.py` restarts
  local subprocesses; it doesn't handle a node itself disappearing (vs. one
  process on it crashing), which needs an external scheduler (Slurm,
  Kubernetes) -- see `MULTI_NODE_DESIGN.md`'s fault-recovery section for
  what that actually requires.
- **The JAX demo is data-parallel only** (`jax.pmap`), the direct JAX
  counterpart to the PyTorch DDP proof -- it doesn't cover JAX's
  `jax.sharding`/GSPMD APIs, which are JAX's answer to FSDP/tensor
  parallelism-style sharding.

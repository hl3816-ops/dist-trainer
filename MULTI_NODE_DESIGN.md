# Scaling dist-trainer beyond one node

Everything in this repo has been built and measured on a single machine (2
GPUs, one process group, one network namespace). This document is a design
analysis, not a validated result: what would actually have to change to run
this `Trainer` across multiple nodes, and why -- written up as a technical
report rather than left as an unstated assumption, since I don't have
multi-node hardware to test it on directly.

## What's already node-count-agnostic

`Trainer` reads `RANK` / `WORLD_SIZE` from the environment and calls
`dist.init_process_group(backend=backend)` with no hardcoded addressing --
it has no concept of "node" at all, only global rank. That's deliberate:
a correctly-configured launcher (below) can hand this exact class 1 node or
100 without a code change. The parts of this repo that *do* hardcode
single-node addressing (`MASTER_ADDR=127.0.0.1` in the Kaggle kernels,
`subprocess.Popen`-launched workers in `tests/test_correctness.py`) are
single-node test/demo scaffolding around the library, not the library
itself.

`checkpoint.py` already assumes "every rank reads the same path" -- true on
one machine's local disk, and *still* true on N nodes as long as that path
is shared storage (NFS, a distributed filesystem, or object storage mounted
consistently). Nothing to change there; the assumption just gets more load-
bearing.

## What would actually have to change

**Launcher and rendezvous.** `torchrun --nproc_per_node=K` on each of N
nodes, with `--nnodes=N --node_rank=R` per node and a `--rdzv_endpoint`
that's a real address reachable from every node -- `127.0.0.1` obviously
stops working the moment there's more than one machine. This also means the
rendezvous host needs to be up and reachable before training starts, and
(per the libuv bug documented in the main README) whatever torch build is
running needs its rendezvous backend actually tested on the target OS/network,
not assumed.

**Network transport.** Intra-node GPU-to-GPU traffic on this project's T4s
goes over PCIe; on real multi-GPU nodes it's usually NVLink (hundreds of
GB/s). Inter-node traffic goes over the network fabric -- InfiniBand if
present (tens of GB/s), plain Ethernet/TCP if not (often an order of
magnitude slower again). NCCL auto-detects and picks the fastest available
transport, but "auto-detects" is exactly the kind of thing that fails
silently in a way worth knowing how to debug: `NCCL_DEBUG=INFO` shows which
transport NCCL actually chose per link, `NCCL_SOCKET_IFNAME` pins it to the
correct NIC when a node has multiple interfaces (management network vs.
data network) and NCCL guesses wrong, and `NCCL_IB_DISABLE=1` forces a
TCP fallback for diagnosing whether an issue is IB-specific.

**Why the bandwidth gap matters, concretely.** DDP and FSDP both need every
rank's gradients (or gradient shards) synchronized every step. Within a
node, that's cheap. Across nodes, an all-reduce spanning multiple nodes is
bottlenecked by whichever link is slowest -- meaning the *inter-node* links,
not the intra-node ones, set the ceiling on scaling efficiency. This repo's
2-GPU FSDP benchmark already shows communication cost rising with model
size on a *single node* (see main README); at multi-node scale that same
effect gets materially worse before any optimization, which is why:

- **Gradient bucketing/overlap** (DDP's `bucket_cap_mb`, tuned rather than
  left at its default) matters more here than it did on 2 GPUs -- starting
  an all-reduce on already-computed gradient buckets while backward is
  still computing later layers hides communication latency behind compute
  that's already happening anyway. The benefit is proportional to how
  expensive each all-reduce is, which is exactly what gets worse crossing
  node boundaries.
- **Topology-aware placement** (keeping tensor-parallel shards, which
  communicate every layer, on GPUs connected by NVLink within one node;
  reserving the slower inter-node links for the comparatively infrequent
  data-parallel gradient sync) is why real large-model training combines
  multiple parallelism strategies instead of just scaling DDP/FSDP flat
  across every GPU regardless of which node it's on.

**Fault tolerance at cluster scale.** This repo's fault-tolerance proof
(`tests/test_fault_tolerance.py`) kills and resumes a single process on one
machine. At hundreds of nodes, *some* node failing during a long run stops
being an edge case and starts being close to guaranteed -- so the recovery
story has to be automatic, not "someone notices and re-runs the command":
`torchrun`'s elastic mode (`--nnodes=MIN:MAX`) or an external
scheduler (Slurm, Kubernetes with health checks) needs to detect the dead
node, remove or replace it, and resume the surviving ranks from the last
checkpoint without manual intervention. Checkpoint *writing* itself needs
attention at this scale too -- serializing a large model's full optimizer
state is easily terabytes, and doing that synchronously (stalling training
until the write finishes) wastes GPU-hours proportional to cluster size;
`torch.distributed.checkpoint`'s sharded, async-capable checkpointing exists
specifically for this, and this repo's simpler "rank 0 blocks and writes a
single file" (`dist_trainer/checkpoint.py`) is a deliberate small-scale
simplification, not something to scale up as-is.

**Data loading.** With one small text file, this was never a bottleneck
here. At real scale, every node needing a full local copy of the dataset
doesn't work (storage, and the time to distribute it), so training data
needs to be shardable/streamable -- each rank reading only its own slice,
with enough read-ahead/prefetch that the GPUs are never idle waiting on I/O.
`examples/gpt_data.py`'s "load the whole tiny file into one tensor" approach
is fine at this project's scale and nowhere close to fine beyond it.

## What I'd want to actually validate, given real multi-node hardware

This document reasons from how NCCL and `torch.distributed` are documented
to behave, not from having measured it -- the honest next step, if the
hardware were available, would be running the same correctness invariant
this repo already proves on 2 GPUs (`tests/test_correctness.py`) across
multiple *nodes*, then profiling (as in `kaggle_kernel/profiling/`) to see
the intra- vs. inter-node communication split directly instead of inferring
it, and only then tuning bucket sizes / NCCL transport settings against
real numbers rather than documented expectations.

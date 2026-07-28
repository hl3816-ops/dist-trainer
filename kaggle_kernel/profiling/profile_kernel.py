"""Self-contained Kaggle kernel, part 3: where does the DDP/FSDP throughput
difference actually come from?

The earlier benchmarks (kaggle_kernel/benchmark_kernel.py,
kaggle_kernel/gpu_extras/) report aggregate before/after numbers -- "FSDP is
N% slower" -- but never show *why*. This kernel wraps a few training steps of
each parallelism strategy in torch.profiler and reports the actual top CUDA
ops by time, splitting out NCCL communication ops (allreduce / all_gather /
reduce_scatter) from compute ops (matmul / softmax / etc), so the
communication-overhead explanation in the README is backed by a real trace,
not an inference from throughput numbers alone.

Self-contained for the same reason as the other kernels in this folder.
"""

import json
import math
import os

# ---------------------------------------------------------------------------
# Data + model (same as the other kernels in this folder)
# ---------------------------------------------------------------------------
import urllib.request

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile


def load_text() -> str:
    cache_path = "/kaggle/working/tinyshakespeare.txt"
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url, timeout=20) as resp:
        text = resp.read().decode("utf-8")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}


def make_batch_iter(data, block_size, total_batch_size, seed=0):
    n = data.shape[0]

    def batch_iter(step, rank, world_size):
        shard_size = total_batch_size // world_size
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        starts = torch.randint(
            0, n - block_size - 1, (total_batch_size,), generator=gen
        )
        my_starts = starts[rank * shard_size : (rank + 1) * shard_size]
        x = torch.stack([data[s : s + block_size] for s in my_starts])
        y = torch.stack([data[s + 1 : s + 1 + block_size] for s in my_starts])
        return x, y

    return batch_iter


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        b, t, c = x.shape
        qkv = (
            self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def loss_fn(logits, targets):
    b, t, v = logits.shape
    return F.cross_entropy(logits.view(b * t, v), targets.view(b * t))


# Bigger model (same "big_300M" config as gpu_extras_kernel.py): the
# communication-vs-compute split is far more visible at this size than at
# 11M params, where compute dominates the trace almost entirely.
N_LAYER, N_HEAD, N_EMBD = 24, 16, 1024
BLOCK_SIZE = 128
TOTAL_BATCH_SIZE = 32
SEED = 0
PROFILE_STEPS = 8
WARMUP_STEPS = 3

_COMM_OP_MARKERS = (
    "allreduce",
    "all_reduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "nccl",
)


def _device_time(e) -> float:
    # torch >=2.1 renamed the profiler's cuda_time_total/self_cuda_time_total
    # to self_device_time_total (generalized beyond CUDA); fall back to the
    # older name for older torch builds.
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        val = getattr(e, attr, 0)
        if val:
            return val
    return 0


def _classify_and_summarize(prof, top_n=12):
    events = prof.key_averages()
    rows = []
    comm_time_us = 0
    total_time_us = 0
    for e in events:
        cuda_time = _device_time(e)
        if cuda_time <= 0:
            continue
        total_time_us += cuda_time
        is_comm = any(marker in e.key.lower() for marker in _COMM_OP_MARKERS)
        if is_comm:
            comm_time_us += cuda_time
        rows.append({"op": e.key, "cuda_time_us": cuda_time, "is_comm": is_comm})

    rows.sort(key=lambda r: r["cuda_time_us"], reverse=True)
    comm_pct = round(100 * comm_time_us / total_time_us, 2) if total_time_us else 0.0
    return {
        "total_cuda_time_us": total_time_us,
        "comm_cuda_time_us": comm_time_us,
        "comm_pct_of_total": comm_pct,
        "top_ops": rows[:top_n],
    }


def _profile_one(rank, world_size, use_fsdp, holder):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29603"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)

    text = load_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor([tokenizer.stoi[c] for c in text], dtype=torch.long)

    model = TinyGPT(tokenizer.vocab_size, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD).to(
        device
    )
    model = FSDP(model) if use_fsdp else DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_iter = make_batch_iter(data, BLOCK_SIZE, TOTAL_BATCH_SIZE, seed=SEED)

    def step(i):
        x, y = batch_iter(i, rank, world_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    for i in range(WARMUP_STEPS):
        step(i)
    torch.cuda.synchronize(device)
    dist.barrier()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(WARMUP_STEPS, WARMUP_STEPS + PROFILE_STEPS):
            step(i)
        torch.cuda.synchronize(device)

    if rank == 0:
        holder["result"] = {
            "parallelism": "fsdp" if use_fsdp else "ddp",
            **_classify_and_summarize(prof),
        }

    dist.destroy_process_group()


def run_profiling():
    results = {}
    for use_fsdp in [False, True]:
        manager = mp.Manager()
        holder = manager.dict()
        mp.spawn(_profile_one, args=(2, use_fsdp, holder), nprocs=2, join=True)
        result = dict(holder["result"])
        label = result["parallelism"]
        print(f"\n=== {label.upper()} ===")
        print(
            f"total CUDA time (profiled steps): {result['total_cuda_time_us'] / 1000:.1f} ms"
        )
        print(
            f"communication ops: {result['comm_pct_of_total']}% of total CUDA time "
            f"({result['comm_cuda_time_us'] / 1000:.1f} ms)"
        )
        print("top ops by CUDA time:")
        for op in result["top_ops"][:8]:
            tag = "[COMM]" if op["is_comm"] else "[compute]"
            print(
                f"  {tag:10s} {op['op'][:60]:60s} {op['cuda_time_us'] / 1000:8.2f} ms"
            )
        results[label] = result

    return results


def main():
    n_gpus = torch.cuda.device_count()
    print(f"CUDA devices available: {n_gpus}")
    if n_gpus < 2:
        print("Need >=2 GPUs for this kernel; skipping.")
        return

    results = run_profiling()

    with open("/kaggle/working/profiling_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n\n=== SUMMARY ===")
    for label, r in results.items():
        print(f"{label}: {r['comm_pct_of_total']}% of CUDA time in communication ops")


if __name__ == "__main__":
    main()

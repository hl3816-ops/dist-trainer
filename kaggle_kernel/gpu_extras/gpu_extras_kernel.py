"""Self-contained Kaggle kernel, part 2: two things the CPU-only tests and
the AMP/checkpoint benchmark didn't cover.

  1. GPU correctness: re-runs the DDP-vs-single-process weight-equivalence
     check (tests/test_correctness.py) on real GPUs with the NCCL backend,
     not just CPU/gloo -- the CPU version proves the mechanics are right,
     this proves they're *also* right on the backend real training actually
     uses.
  2. FSDP vs. DDP: same task, same step count, 2 GPUs -- DDP replicates the
     full model/optimizer state on every rank; FSDP shards it. Compares
     peak memory and throughput to show the actual effect of sharding,
     not just that FSDP "runs".

Self-contained for the same reason as benchmark_kernel.py: a single script
is the simplest reliable Kaggle push. See dist_trainer/trainer.py for the
reusable library version (Trainer(..., parallelism="fsdp")).
"""

import json
import math
import os
import time
import urllib.request

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Data + model (same as benchmark_kernel.py)
# ---------------------------------------------------------------------------

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text() -> str:
    cache_path = "/kaggle/working/tinyshakespeare.txt"
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()
    with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=20) as resp:
        text = resp.read().decode("utf-8")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


class CharTokenizer:
    def __init__(self, text: str):
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


STEPS = 30
WARMUP = 5
TOTAL_BATCH_SIZE = 64
BLOCK_SIZE = 128
N_LAYER, N_HEAD, N_EMBD = 6, 6, 384  # used by the correctness check ("small")
SEED = 0

# Model sizes for the FSDP-vs-DDP comparison, to show the trend (not just one
# data point): "small" is the same ~11M-param config used everywhere else in
# this repo; "big" is as large as comfortably fits DDP's full-replica memory
# on a single 16GB T4 (~300M params, GPT-2-small/medium-ish depth/width).
# Still nowhere near an industry "large model" -- see README for that caveat.
MODEL_CONFIGS = {
    "small_11M": {"n_layer": 6, "n_head": 6, "n_embd": 384, "total_batch_size": 64},
    "big_300M": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "total_batch_size": 32},
}


def _load_data():
    text = load_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor([tokenizer.stoi[c] for c in text], dtype=torch.long)
    return tokenizer, data


# ---------------------------------------------------------------------------
# Part 1: GPU correctness (DDP vs single-process, NCCL)
# ---------------------------------------------------------------------------


def _train_and_get_state(rank, world_size, device, holder, wrap_ddp):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29601"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    tokenizer, data = _load_data()

    model = TinyGPT(tokenizer.vocab_size, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD).to(
        device
    )
    if wrap_ddp:
        model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    batch_iter = make_batch_iter(data, BLOCK_SIZE, TOTAL_BATCH_SIZE, seed=SEED)

    for i in range(STEPS):
        x, y = batch_iter(i, rank, world_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    if rank == 0:
        underlying = model.module if hasattr(model, "module") else model
        holder["state"] = {
            k: v.detach().cpu() for k, v in underlying.state_dict().items()
        }

    if world_size > 1:
        dist.destroy_process_group()


def _correctness_spawn_entry(rank, world_size, holder):
    local_holder = {}
    _train_and_get_state(
        rank, world_size, torch.device(f"cuda:{rank}"), local_holder, wrap_ddp=True
    )
    if rank == 0:
        holder["state"] = local_holder["state"]


def run_gpu_correctness_check():
    single_holder = {}
    _train_and_get_state(0, 1, torch.device("cuda:0"), single_holder, wrap_ddp=False)
    single_state = single_holder["state"]

    manager = mp.Manager()
    multi_holder = manager.dict()
    mp.spawn(_correctness_spawn_entry, args=(2, multi_holder), nprocs=2, join=True)
    multi_state = dict(multi_holder["state"])

    max_diff = 0.0
    for name in single_state:
        diff = (single_state[name] - multi_state[name]).abs().max().item()
        max_diff = max(max_diff, diff)

    result = {
        "max_abs_weight_diff": max_diff,
        "passed": max_diff < 1e-3,
        "backend": "nccl",
        "steps": STEPS,
    }
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Part 2: FSDP vs DDP (peak memory + throughput)
# ---------------------------------------------------------------------------


def _count_params(n_layer, n_head, n_embd, vocab_size, block_size):
    m = TinyGPT(vocab_size, block_size, n_layer, n_head, n_embd)
    return sum(p.numel() for p in m.parameters())


def _run_parallel_config(rank, world_size, use_fsdp, model_cfg, holder):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29602"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)
    tokenizer, data = _load_data()

    n_layer, n_head, n_embd = (
        model_cfg["n_layer"],
        model_cfg["n_head"],
        model_cfg["n_embd"],
    )
    total_batch_size = model_cfg["total_batch_size"]

    model = TinyGPT(tokenizer.vocab_size, BLOCK_SIZE, n_layer, n_head, n_embd).to(
        device
    )
    model = FSDP(model) if use_fsdp else DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_iter = make_batch_iter(data, BLOCK_SIZE, total_batch_size, seed=SEED)

    def step(i):
        x, y = batch_iter(i, rank, world_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    for i in range(WARMUP):
        step(i)
    torch.cuda.synchronize(device)
    dist.barrier()

    start = time.perf_counter()
    for i in range(WARMUP, WARMUP + STEPS):
        step(i)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        samples_per_sec = (STEPS * total_batch_size) / elapsed
        holder["result"] = {
            "parallelism": "fsdp" if use_fsdp else "ddp",
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "peak_cuda_memory_mb": round(
                torch.cuda.max_memory_allocated(device) / 1e6, 1
            ),
        }

    dist.destroy_process_group()


def run_fsdp_vs_ddp():
    all_results = {}
    for label, model_cfg in MODEL_CONFIGS.items():
        num_params = _count_params(
            model_cfg["n_layer"],
            model_cfg["n_head"],
            model_cfg["n_embd"],
            65,
            BLOCK_SIZE,
        )
        print(f"\n--- model config {label} (~{num_params / 1e6:.1f}M params) ---")
        results = []
        for use_fsdp in [False, True]:
            manager = mp.Manager()
            holder = manager.dict()
            mp.spawn(
                _run_parallel_config,
                args=(2, use_fsdp, model_cfg, holder),
                nprocs=2,
                join=True,
            )
            result = dict(holder["result"])
            result["num_params_approx"] = num_params
            print(json.dumps(result, indent=2))
            results.append(result)
        all_results[label] = results
    return all_results


def main():
    n_gpus = torch.cuda.device_count()
    print(f"CUDA devices available: {n_gpus}")
    if n_gpus < 2:
        print(
            "Need >=2 GPUs for this kernel; skipping (see benchmark_kernel.py for 1-GPU results)."
        )
        return

    print("\n=== Part 1: GPU correctness (DDP vs single-process, NCCL) ===")
    correctness_result = run_gpu_correctness_check()

    print("\n=== Part 2: FSDP vs DDP (2 GPUs) ===")
    fsdp_results = run_fsdp_vs_ddp()

    output = {"gpu_correctness": correctness_result, "fsdp_vs_ddp": fsdp_results}
    with open("/kaggle/working/gpu_extras_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\n\n=== SUMMARY ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

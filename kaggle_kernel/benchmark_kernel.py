"""Self-contained Kaggle kernel: runs the dist-trainer TinyGPT throughput/
memory benchmark across a matrix of (world_size, use_amp,
use_grad_checkpoint) configs on real GPU hardware (NCCL backend), and prints
a before/after comparison table.

This inlines the model/data/training-step code from
https://github.com/hl3816-ops/dist-trainer rather than importing the repo,
since a single self-contained script is the simplest reliable way to push a
Kaggle kernel. See that repo for the reusable library version (with the
full Trainer class, checkpointing, and the CPU correctness/fault-tolerance
tests this benchmark builds on).

Multi-GPU runs use torch.multiprocessing.spawn directly rather than
`torchrun`, so this needs no subprocess/launcher setup at all -- just run
the script.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Data: character-level tiny-shakespeare
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


def make_batch_iter(
    data: torch.Tensor, block_size: int, total_batch_size: int, seed: int = 0
):
    n = data.shape[0]

    def batch_iter(step: int, rank: int, world_size: int):
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


# ---------------------------------------------------------------------------
# Model: minimal decoder-only transformer (see examples/gpt_model.py)
# ---------------------------------------------------------------------------


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
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embd,
        use_gradient_checkpointing,
    ):
        super().__init__()
        self.block_size = block_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
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

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(self.ln_f(x))


def loss_fn(logits, targets):
    b, t, v = logits.shape
    return F.cross_entropy(logits.view(b * t, v), targets.view(b * t))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

STEPS = 40
WARMUP = 10
TOTAL_BATCH_SIZE = 64
BLOCK_SIZE = 128
N_LAYER, N_HEAD, N_EMBD = 6, 6, 384
SEED = 0


def run_one(rank, world_size, use_amp, use_grad_checkpoint, result_holder):
    is_distributed = world_size > 1
    if is_distributed:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29600"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)

    text = load_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor([tokenizer.stoi[c] for c in text], dtype=torch.long)

    model = TinyGPT(
        tokenizer.vocab_size, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, use_grad_checkpoint
    )
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_iter = make_batch_iter(data, BLOCK_SIZE, TOTAL_BATCH_SIZE, seed=SEED)

    def step(i):
        x, y = batch_iter(i, rank, world_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    for i in range(WARMUP):
        step(i)
    torch.cuda.synchronize(device)
    if is_distributed:
        dist.barrier()

    start = time.perf_counter()
    for i in range(WARMUP, WARMUP + STEPS):
        step(i)
    torch.cuda.synchronize(device)
    if is_distributed:
        dist.barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        samples_per_sec = (STEPS * TOTAL_BATCH_SIZE) / elapsed
        result_holder["result"] = {
            "world_size": world_size,
            "use_amp": use_amp,
            "use_grad_checkpoint": use_grad_checkpoint,
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "tokens_per_sec": round(samples_per_sec * BLOCK_SIZE, 1),
            "peak_cuda_memory_mb": round(
                torch.cuda.max_memory_allocated(device) / 1e6, 1
            ),
        }

    if is_distributed:
        dist.destroy_process_group()


def run_config(world_size, use_amp, use_grad_checkpoint):
    if world_size == 1:
        holder = {}
        run_one(0, 1, use_amp, use_grad_checkpoint, holder)
        return holder["result"]

    manager = mp.Manager()
    holder = manager.dict()
    mp.spawn(
        _spawn_entry,
        args=(world_size, use_amp, use_grad_checkpoint, holder),
        nprocs=world_size,
        join=True,
    )
    return dict(holder["result"])


def _spawn_entry(rank, world_size, use_amp, use_grad_checkpoint, holder):
    local_holder = {}
    run_one(rank, world_size, use_amp, use_grad_checkpoint, local_holder)
    if rank == 0:
        holder["result"] = local_holder["result"]


def main():
    n_gpus = torch.cuda.device_count()
    print(f"CUDA devices available: {n_gpus}")
    world_sizes = [1, 2] if n_gpus >= 2 else [1]

    results = []
    for world_size in world_sizes:
        for use_amp in [False, True]:
            for use_grad_checkpoint in [False, True]:
                print(
                    f"\n=== world_size={world_size} amp={use_amp} grad_ckpt={use_grad_checkpoint} ==="
                )
                result = run_config(world_size, use_amp, use_grad_checkpoint)
                print(json.dumps(result, indent=2))
                results.append(result)

    print("\n\n=== SUMMARY ===")
    print(
        f"{'world_size':>10} {'amp':>6} {'grad_ckpt':>10} "
        f"{'samples/s':>10} {'tokens/s':>10} {'peak_mem_mb':>12}"
    )
    for r in results:
        print(
            f"{r['world_size']:>10} {r['use_amp']!s:>6} {r['use_grad_checkpoint']!s:>10} "
            f"{r['samples_per_sec']:>10} {r['tokens_per_sec']:>10} {r['peak_cuda_memory_mb']:>12}"
        )

    with open("/kaggle/working/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

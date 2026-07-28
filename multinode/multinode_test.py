"""Real 2-NODE (not 2-process-on-1-node) DDP correctness test, run across
two separate RunPod machines bridged by an SSH tunnel relay for the NCCL
rendezvous. Each node runs this script once, with RANK/WORLD_SIZE/
MASTER_ADDR/MASTER_PORT set via environment by the driving SSH command.

Rank 0 also trains a single-process reference (before touching distributed
at all) so the comparison is self-contained in one run.
"""

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

BLOCK_SIZE = 64
N_LAYER, N_HEAD, N_EMBD = 4, 4, 256
TOTAL_BATCH_SIZE = 16
STEPS = 20
SEED = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
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
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
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


VOCAB_SIZE = 65


def make_batch_iter(total_batch_size, seed=0):
    def batch_iter(step, rank, world_size):
        shard_size = total_batch_size // world_size
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        x_full = torch.randint(0, VOCAB_SIZE, (total_batch_size, BLOCK_SIZE), generator=gen)
        y_full = torch.randint(0, VOCAB_SIZE, (total_batch_size, BLOCK_SIZE), generator=gen)
        start = rank * shard_size
        return x_full[start : start + shard_size], y_full[start : start + shard_size]

    return batch_iter


def loss_fn(logits, targets):
    b, t, v = logits.shape
    return F.cross_entropy(logits.view(b * t, v), targets.view(b * t))


def train(world_size, rank, wrap_ddp, device):
    torch.manual_seed(SEED)
    model = TinyGPT(VOCAB_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD).to(device)
    if wrap_ddp:
        model = DDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    batch_iter = make_batch_iter(TOTAL_BATCH_SIZE, seed=SEED)

    for i in range(STEPS):
        x, y = batch_iter(i, rank, world_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    underlying = model.module if hasattr(model, "module") else model
    return {k: v.detach().cpu() for k, v in underlying.state_dict().items()}


def main():
    import datetime

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda:0")

    if rank == 0:
        print("Rank 0: training single-process reference (pre-distributed)...", flush=True)
        single_state = train(1, 0, wrap_ddp=False, device=device)
        torch.save(single_state, "/tmp/single_reference.pt")
        print("Rank 0: reference done.", flush=True)

    print(f"[rank {rank}] initializing gloo process group, world_size={world_size}...", flush=True)
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=60))
    print(f"[rank {rank}] process group initialized OK", flush=True)

    multi_state = train(world_size, rank, wrap_ddp=True, device=device)
    print(f"[rank {rank}] training done", flush=True)

    if rank == 0:
        single_state = torch.load("/tmp/single_reference.pt", weights_only=True)
        max_diff = 0.0
        for name in single_state:
            diff = (single_state[name] - multi_state[name]).abs().max().item()
            max_diff = max(max_diff, diff)
        result = {
            "world_size": world_size,
            "max_abs_weight_diff": max_diff,
            "passed": max_diff < 1e-3,
            "backend": "gloo",
            "topology": "2 real separate nodes (RunPod), bridged via SSH tunnel relay",
        }
        print("RESULT_JSON:" + json.dumps(result), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

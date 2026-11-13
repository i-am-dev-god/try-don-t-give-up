"""The same tiny GPT, but in PyTorch, so you can see the scale gap for yourself.

nn.py trains a ~10k-parameter transformer on 40 characters of text in seconds
on a laptop CPU. GPT-class models have ~10k times more parameters AND hit
every trick this file doesn't: multi-GPU training, mixed precision, gradient
checkpointing, flash attention, 10+ trillion tokens.

Run:
    python torch_gpt.py                 # tiny demo, prints loss going down
    python torch_gpt.py --params 10e6   # sanity check on model size only
"""

from __future__ import annotations

import argparse
import math
import time


def build_torch_model(vocab: int, dim: int, heads: int, blocks: int, ctx: int):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        def forward(self, x, mask):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, dim)
            self.pos = nn.Embedding(ctx, dim)
            self.blocks = nn.ModuleList([Block() for _ in range(blocks)])
            self.ln_f = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)
            self.ctx = ctx

        def forward(self, ids):
            t = ids.shape[1]
            x = self.tok(ids) + self.pos(torch.arange(t, device=ids.device))
            mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
            for b in self.blocks:
                x = b(x, mask)
            return self.head(self.ln_f(x))

    return GPT()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=0, help="just report a model this big")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed here. That is fine: nn.py is the point.")
        print("  pip install torch   # then re-run for the side-by-side")
        return

    if args.params:
        print(f"a model with {args.params:,.0f} parameters is "
              f"{args.params / 10_000:,.0f}x the size of the from-scratch demo.")
        return

    text = ("the quick brown fox jumps over the lazy dog. " * 40)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text])
    ctx = 16
    model = build_torch_model(len(vocab), args.dim, args.heads, args.blocks, ctx)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    windows = torch.stack([ids[i:i + ctx + 1] for i in range(len(ids) - ctx)])
    t0 = time.time()
    for step in range(args.steps):
        x, y = windows[:, :-1], windows[:, 1:]
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:3d}  loss {loss.item():.3f}")
    print(f"  {n:,} params, {args.steps} steps in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

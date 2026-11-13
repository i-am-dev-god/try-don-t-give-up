"""Train a real (tiny) GPT from scratch, in about two seconds, on a CPU.

This is the same code path as models.TinyGPT: token + position embeddings,
causal self-attention, residual blocks, a language-model head, and
next-character prediction with cross-entropy. The only thing shrunk is scale.

Run:
    python mini_gpt.py                     # trains on this README
    python mini_gpt.py --file finance.py   # or any other text file
    python mini_gpt.py --steps 400 --dim 64  # longer, better samples

Watch the loss. It starts around ln(vocab) -- pure guessing -- and falls as the
model picks up letter frequencies, then words, then the rhythm of the source.
That curve, at 10,000x the data and 100,000x the parameters, is what training
GPT looks like.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from models import TinyGPT, cross_entropy
from nn import Adam, count_params

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(HERE, "README.md"))
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = open(args.file, encoding="utf-8").read()
    if len(text) < 2000:
        text = (text + "\n") * (2000 // max(1, len(text)) + 1)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text])

    rng = np.random.default_rng(args.seed)
    model = TinyGPT(len(vocab), args.dim, args.heads, args.blocks, args.context, rng)
    print(f"{len(text):,} characters, {len(vocab)} distinct, "
          f"{count_params(model):,} parameters")
    print(f"{args.blocks} block(s), {args.heads} heads, context {args.context}")

    opt = Adam(model.params(), lr=args.lr)
    starts = np.arange(len(text) - args.context - 1)
    t0 = time.time()
    for step in range(args.steps):
        idx = rng.choice(starts, args.batch, replace=False)
        x = np.stack([ids[i:i + args.context] for i in idx])
        y = np.stack([ids[i + 1:i + args.context + 1] for i in idx]).reshape(-1)
        loss = cross_entropy(model(x), y)
        model.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {float(loss.data):.3f}  "
                  f"({time.time() - t0:.1f}s)")

    # greedy-ish sampling from the trained model
    prompt = text[:12]
    out = model.generate([stoi.get(c, 0) for c in prompt], 120, rng, temperature=0.7)
    print("\n--- sample ---")
    print("".join(vocab[i] for i in out))
    print("--------------")
    print("\nIt writes nonsense, but the nonsense has the *texture* of the source:")
    print("punctuation, line breaks, word-shaped chunks. That is a language model.")
    print(f"GPT-class models have ~10,000x more parameters and ~10^12x more data.")


if __name__ == "__main__":
    main()

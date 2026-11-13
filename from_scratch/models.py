"""Three models, all built from the Tensor ops in nn.py.

  MLP      - the "hello world" of neural nets: layers of linear + nonlinearity.
  RNN      - a recurrent net that reads a sequence, carrying a hidden state.
  TinyGPT  - a decoder-only transformer with causal self-attention: the same
             architecture family as GPT, shrunk until it trains on a CPU.
"""

from __future__ import annotations

import numpy as np

from nn import (Embedding, LayerNorm, Linear, Module, Tensor, log_softmax,
                matmul, mean, mul, power, relu, softmax, tanh)


# ---------------------------------------------------------------------------
# Small shape ops that autograd can flow through
# ---------------------------------------------------------------------------


def _reshape(x: Tensor, shape, back_shape) -> Tensor:
    out = Tensor(x.data.reshape(shape), requires_grad=x.requires_grad)
    if x.requires_grad:
        def _back():
            g = out.grad.reshape(back_shape)
            x.grad = g if x.grad is None else x.grad + g
        out._parents = (x,)
        out._backward = _back
    return out


def _permute(x: Tensor, axes) -> Tensor:
    out = Tensor(np.transpose(x.data, axes), requires_grad=x.requires_grad)
    inv = tuple(np.argsort(axes))
    if x.requires_grad:
        def _back():
            g = np.transpose(out.grad, inv)
            x.grad = g if x.grad is None else x.grad + g
        out._parents = (x,)
        out._backward = _back
    return out


def _split_last(x: Tensor, n: int) -> list[Tensor]:
    """Slice the last dim into n equal chunks, accumulating grads back into x."""
    d = x.shape[-1] // n
    parts = []
    for i in range(n):
        sl = x.data[..., i * d:(i + 1) * d]
        out = Tensor(sl, requires_grad=x.requires_grad)
        if x.requires_grad:
            # bind loop vars as defaults: closures capture names, not values
            def _back(i=i, out=out):
                g = np.zeros_like(x.data)
                g[..., i * d:(i + 1) * d] = out.grad
                x.grad = g if x.grad is None else x.grad + g
            out._parents = (x,)
            out._backward = _back
        parts.append(out)
    return parts


# ---------------------------------------------------------------------------
# MLP: features -> hidden -> prediction
# ---------------------------------------------------------------------------


class MLP(Module):
    def __init__(self, d_in: int, hidden: int, d_out: int, rng, depth: int = 2):
        dims = [d_in] + [hidden] * depth
        self.layers = [Linear(dims[i], dims[i + 1], rng) for i in range(depth)]
        self.head = Linear(hidden, d_out, rng)

    def forward(self, x):
        for layer in self.layers:
            x = relu(layer(x))
        return self.head(x)

    __call__ = forward


# ---------------------------------------------------------------------------
# RNN: read a window of days one at a time, remembering a summary
# ---------------------------------------------------------------------------


class RNN(Module):
    """Elman RNN:  h_t = tanh(x_t Wx + h_{t-1} Wh + b)  ->  linear head.

    The loop over time is what makes it "recurrent"; backprop-through-time is
    just autograd unrolling the very same graph.
    """

    def __init__(self, d_in: int, hidden: int, d_out: int, rng):
        self.Wx = Linear(d_in, hidden, rng, bias=False)
        self.Wh = Linear(hidden, hidden, rng)          # its bias is b
        self.head = Linear(hidden, d_out, rng)
        self.hidden = hidden

    def forward(self, x):
        # x: (batch, time, d_in)
        batch, steps, _ = x.shape
        h = Tensor(np.zeros((batch, self.hidden)))
        for t in range(steps):
            h = tanh(self.Wx(x[:, t, :]) + self.Wh(h))
        return self.head(h)

    __call__ = forward


# ---------------------------------------------------------------------------
# TinyGPT
# ---------------------------------------------------------------------------


def causal_mask(steps: int):
    """Upper-triangular -inf: position t may only attend to positions <= t."""
    return np.triu(np.full((1, 1, steps, steps), -1e9), k=1)


def attention(q, k, v, mask, n_heads: int):
    """Scaled dot-product attention with multiple heads.

    q, k, v : (batch, time, dim)
    1. split the channel dim into n_heads, giving (batch*n_heads, time, d/heads)
    2. scores = q k^T / sqrt(d_head), mask out the future, softmax
    3. output  = scores @ v, then merge heads back into (batch, time, dim)
    """
    b, t, d = q.shape
    hd = d // n_heads

    def heads(x):
        # (b, t, heads, hd) -> (b, heads, t, hd) -> (b*heads, t, hd)
        return _reshape(_permute(_reshape(x, (b, t, n_heads, hd), (b, t, d)),
                                 (0, 2, 1, 3)),
                        (b * n_heads, t, hd), (b, n_heads, t, hd))

    qh, kh, vh = heads(q), heads(k), heads(v)
    scores = matmul(qh, _permute(kh, (0, 2, 1))) * (1.0 / np.sqrt(hd))
    weights = softmax(scores + mask, axis=-1)
    out = matmul(weights, vh)
    # (b*heads, t, hd) -> (b, heads, t, hd) -> (b, t, heads, hd) -> (b, t, d)
    merged = _reshape(_permute(_reshape(out, (b, n_heads, t, hd), (b * n_heads, t, hd)),
                               (0, 2, 1, 3)),
                      (b, t, d), (b, t, n_heads, hd))
    return merged


class Block(Module):
    """Transformer block: attention then MLP, both added into a residual stream."""

    def __init__(self, dim: int, n_heads: int, rng, expansion: int = 4):
        self.dim = dim
        self.n_heads = n_heads
        self.ln1 = LayerNorm(dim)
        self.qkv = Linear(dim, 3 * dim, rng)
        self.proj = Linear(dim, dim, rng)
        self.ln2 = LayerNorm(dim)
        self.fc1 = Linear(dim, expansion * dim, rng)
        self.fc2 = Linear(expansion * dim, dim, rng)

    def forward(self, x, mask):
        q, k, v = _split_last(self.qkv(self.ln1(x)), 3)
        x = x + self.proj(attention(q, k, v, mask, self.n_heads))
        return x + self.fc2(relu(self.fc1(self.ln2(x))))

    __call__ = forward


class TinyGPT(Module):
    """Decoder-only transformer trained to predict the next character."""

    def __init__(self, vocab: int, dim: int, n_heads: int, n_blocks: int,
                 context: int, rng):
        self.context = context
        self.tok = Embedding(vocab, dim, rng)
        self.pos = Embedding(context, dim, rng)
        self.blocks = [Block(dim, n_heads, rng) for _ in range(n_blocks)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab, rng, bias=False)

    def forward(self, ids):
        b, t = ids.shape
        x = self.tok(ids) + self.pos(np.arange(t))
        mask = causal_mask(t)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))                   # (b, t, vocab)

    __call__ = forward

    def generate(self, prompt_ids, n_new: int, rng, temperature: float = 0.8):
        ids = list(prompt_ids)
        for _ in range(n_new):
            window = np.array(ids[-self.context:])[None, :]
            logits = self.forward(window).data[0, -1] / max(temperature, 1e-6)
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            ids.append(int(rng.choice(len(probs), p=probs)))
        return ids


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Average negative log-likelihood of the correct next token."""
    b, t, v = logits.shape
    flat = _reshape(logits, (b * t, v), (b, t, v))
    logp = log_softmax(flat, axis=-1)
    picked = _gather(logp, targets.reshape(-1))
    return mean(picked) * -1.0          # negative log-likelihood


def _gather(logp: Tensor, targets: np.ndarray) -> Tensor:
    idx = targets.reshape(-1, 1)
    out = Tensor(np.take_along_axis(logp.data, idx, axis=1).reshape(-1),
                 requires_grad=logp.requires_grad)
    if logp.requires_grad:
        def _back():
            g = np.zeros_like(logp.data)
            np.put_along_axis(g, idx, out.grad.reshape(-1, 1), axis=1)
            logp.grad = g if logp.grad is None else logp.grad + g
        out._parents = (logp,)
        out._backward = _back
    return out

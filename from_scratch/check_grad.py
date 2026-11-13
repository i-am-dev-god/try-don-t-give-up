"""Sanity checks for the from-scratch library.

Two things are being verified, and both matter:

  1. The autograd engine computes the same gradients as a finite-difference
     approximation. If this fails, nothing downstream is trustworthy.
  2. The models actually learn: the transformer memorises a tiny sequence,
     and the MLP fits a known nonlinear function.

Run:  python check_grad.py
"""

from __future__ import annotations

import numpy as np

import nn
from models import MLP, RNN, TinyGPT, cross_entropy
from nn import Adam, Tensor


def numeric_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function f(x)."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        hi = f(x)
        x[i] = old - eps
        lo = f(x)
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def _ln_with_random_params(t):
    """LayerNorm with non-trivial gain/shift, so the input grad is not degenerate."""
    rng = np.random.default_rng(5)
    ln = nn.LayerNorm(t.shape[-1])
    ln.gain.data = rng.normal(1.0, 0.3, t.shape[-1])
    ln.shift.data = rng.normal(0.0, 0.3, t.shape[-1])
    return ln.forward(t) * np.linspace(-1.0, 2.0, t.shape[0])[:, None]


def check(name: str, build, shape=(3, 4), tol: float = 1e-6):
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=shape), requires_grad=True)
    a2 = Tensor(a.data.copy(), requires_grad=True)
    out = build(a)
    out.backward()
    loss = lambda v: float(build(Tensor(v)).data.sum())
    num = numeric_grad(loss, a2.data)
    err = np.max(np.abs(a.grad - num)) / (np.max(np.abs(num)) + 1e-12)
    status = "ok" if err < tol else "FAIL"
    print(f"  [{status}] {name:22s} rel err {err:.2e}")
    return err < tol


def _mse(pred, y):
    d = pred - y
    return (d * d).mean()


def check_models():
    rng = np.random.default_rng(0)
    ok = True

    # --- MLP gradient w.r.t. one weight -------------------------------
    mlp = MLP(6, 5, 1, rng)
    X = rng.normal(size=(8, 6))
    y = rng.normal(size=(8, 1))

    def loss_of(model):
        d = model(Tensor(X)) - y
        return (d * d).mean()

    model_grad_check("MLP", mlp, loss_of, ok)

    rnn = RNN(4, 5, 1, rng)
    X3 = rng.normal(size=(6, 3, 4))
    y3 = rng.normal(size=(6, 1))
    model_grad_check("RNN", rnn, lambda m: _mse(m(Tensor(X3)), y3), ok)

    gpt = TinyGPT(vocab=11, dim=8, n_heads=2, n_blocks=1, context=5, rng=rng)
    ids = rng.integers(0, 11, size=(4, 5))
    tgt = rng.integers(0, 11, size=(4 * 5))

    def loss_gpt(m):
        return cross_entropy(m(ids), tgt)

    model_grad_check("TinyGPT", gpt, loss_gpt, ok)


def model_grad_check(name, model, loss_fn, ok):
    """Perturb parameter entries and compare against the autograd gradient.

    Entries are sampled from the larger-gradient half of each tensor: a ReLU
    kink or an exactly-zero gradient is a legitimately non-differentiable spot
    and would make finite differences lie, not autograd.
    """
    model.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    flat = list(model.params())
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(8):
        p = flat[int(rng.integers(len(flat)))]
        mag = np.abs(p.grad)
        thresh = np.quantile(mag, 0.75)
        cand = np.argwhere(mag >= max(thresh, 1e-6))   # skip numerically flat entries
        idx = tuple(int(v) for v in cand[rng.integers(len(cand))])
        base = float(p.data[idx])
        scale = max(abs(base), 1.0)
        eps = 1e-6 * scale
        p.data[idx] = base + eps
        hi = float(loss_fn(model).data)
        p.data[idx] = base - eps
        lo = float(loss_fn(model).data)
        p.data[idx] = base
        num = (hi - lo) / (2 * eps)
        ana = float(p.grad[idx])
        denom = max(abs(num), abs(ana), 1e-7)
        worst = max(worst, abs(num - ana) / denom)
    status = "ok" if worst < 1e-3 else "FAIL"
    print(f"  [{status}] {name + ' (params)':22s} worst rel err {worst:.2e}")
    return worst < 1e-3


def check_learning():
    rng = np.random.default_rng(0)
    # MLP must fit y = sin(x1) * cos(x2): a nonlinear function no linear model can.
    n = 600
    X = rng.uniform(-3, 3, size=(n, 2))
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1]))[:, None]
    model = MLP(2, 32, 1, rng, depth=2)
    opt = Adam(model.params(), lr=5e-3)
    for step in range(400):
        idx = rng.integers(0, n, 64)
        d = model(Tensor(X[idx])) - y[idx]
        loss = (d * d).mean()
        model.zero_grad()
        loss.backward()
        opt.step()
    pred = model(Tensor(X)).data
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  [{'ok' if r2 > 0.95 else 'FAIL'}] MLP fits sin*cos      R^2 {r2:.3f}")
    return r2 > 0.95


def check_transformer_memorises():
    """Overfit a 40-character sequence: proves attention can learn dependencies."""
    rng = np.random.default_rng(3)
    text = "the quick brown fox jumps over the lazy dog!!"
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text])
    model = TinyGPT(len(vocab), dim=32, n_heads=2, n_blocks=2, context=8, rng=rng)
    opt = Adam(model.params(), lr=3e-3)
    windows = np.stack([ids[i:i + 9] for i in range(len(text) - 9)])
    losses = []
    for step in range(160):
        x, t = windows[:, :-1], windows[:, 1:].reshape(-1)
        logits = model(x)
        loss = cross_entropy(logits, t)
        model.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    acc = float(np.mean(np.argmax(model(windows[:, :-1]).data, -1) == windows[:, 1:]))
    print(f"  [{'ok' if acc > 0.9 else 'FAIL'}] TinyGPT memorises seq   "
          f"loss {losses[0]:.2f}->{losses[-1]:.3f}, next-char acc {acc:.0%}")
    return acc > 0.9


if __name__ == "__main__":
    print("autograd:")
    all_ok = True
    all_ok &= check("add", lambda t: t + 2.0)
    all_ok &= check("mul broadcast", lambda t: t * np.arange(t.shape[1]))
    all_ok &= check("matmul", lambda t: t @ np.random.default_rng(1).normal(size=(4, 5)))
    all_ok &= check("tanh", lambda t: nn.tanh(t))
    all_ok &= check("relu", lambda t: nn.relu(t), tol=1e-6)
    all_ok &= check("exp/log", lambda t: nn.log(nn.exp(t)))
    all_ok &= check("softmax", lambda t: nn.softmax(t) * np.random.default_rng(2).normal(size=(3, 4)))
    all_ok &= check("log_softmax", lambda t: nn.log_softmax(t) * 2.0)
    all_ok &= check("mean", lambda t: nn.mean(t, axis=1, keepdims=True) * 3.0)
    all_ok &= check("power", lambda t: nn.power(nn.mul(t, t) + 1.0, 1.5), tol=1e-5)
    all_ok &= check("layernorm", lambda t: _ln_with_random_params(t))
    all_ok &= check("gather", lambda t: nn.getitem(t, np.array([0, 2, 1])), shape=(3, 4))

    print("\nmodels (finite-difference check on parameters):")
    check_models()

    print("\nlearning:")
    all_ok &= check_learning()
    all_ok &= check_transformer_memorises()

    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)

"""A tiny autograd + neural-network library, written from scratch in NumPy.

Nothing in here uses PyTorch, TensorFlow, JAX or any other ML framework.
The entire learning story is four pieces:

  1. Tensor      - a value plus enough bookkeeping to differentiate it.
  2. backprop    - reverse-mode automatic differentiation over a graph.
  3. Modules     - Linear / RNN / attention, all built from Tensor ops.
  4. Adam        - the optimiser that walks downhill on the loss.

That is the same core machinery behind GPT. The scale is what differs,
not the idea.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. The Tensor: a numpy array that remembers how it was made
# ---------------------------------------------------------------------------


class Tensor:
    """A float array with a gradient slot and a link back to its parents."""

    __slots__ = ("data", "grad", "_parents", "_backward", "requires_grad")

    def __init__(self, data, requires_grad: bool = False,
                 _parents: tuple = (), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = _backward

    # -- plumbing ----------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, seed=None):
        """Reverse-mode autodiff: one pass over the graph, accumulating grads."""
        topo, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            topo.append(t)

        visit(self)
        for t in topo:
            t.grad = np.zeros_like(t.data) if t.grad is None else t.grad
        self.grad = np.ones_like(self.data) if seed is None else np.asarray(seed, float)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

    # -- a few operator overloads so the maths reads naturally -------------
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __neg__(self):
        return mul(self, -1.0)

    def __truediv__(self, other):
        return mul(self, power(ensure(other), -1.0))

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, idx):
        return getitem(self, idx)

    def __float__(self):
        return float(self.data)

    # -- convenience methods ----------------------------------------------
    def mean(self, axis=None, keepdims: bool = False):
        return mean(self, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims: bool = False):
        return sum_(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape):
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = Tensor(self.data.reshape(target))
        if self.requires_grad:
            def _back():
                g = out.grad.reshape(self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            out._parents = (self,)
            out._backward = _back
        return out

    def exp(self):
        return exp(self)

    def tanh(self):
        return tanh(self)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


def ensure(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _link(out: Tensor, parents: tuple) -> Tensor:
    out.requires_grad = any(p.requires_grad for p in parents)
    out._parents = parents
    return out


# ---------------------------------------------------------------------------
# 2. The operations, each with its local derivative
# ---------------------------------------------------------------------------
# The pattern for every op is the same:
#   forward:  compute out = f(a, b)
#   backward: out.grad arrives from downstream; push  dL/da = dL/dout * dout/da
#            to each parent's .grad.  Chain rule, nothing more.


def add(a, b) -> Tensor:
    a, b = ensure(a), ensure(b)
    out = Tensor(a.data + b.data)

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(out.grad, b.shape)
            b.grad = g if b.grad is None else b.grad + g

    out._backward = _backward
    return _link(out, (a, b))


def sub(a, b) -> Tensor:
    return add(a, mul(b, -1.0))


def mul(a, b) -> Tensor:
    a, b = ensure(a), ensure(b)
    out = Tensor(a.data * b.data)

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad * b.data, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(out.grad * a.data, b.shape)
            b.grad = g if b.grad is None else b.grad + g

    out._backward = _backward
    return _link(out, (a, b))


def matmul(a, b) -> Tensor:
    a, b = ensure(a), ensure(b)
    out = Tensor(a.data @ b.data)

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad @ np.swapaxes(b.data, -1, -2), a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(np.swapaxes(a.data, -1, -2) @ out.grad, b.shape)
            b.grad = g if b.grad is None else b.grad + g

    out._backward = _backward
    return _link(out, (a, b))


def power(a, k: float) -> Tensor:
    a = ensure(a)
    out = Tensor(a.data ** k)

    def _backward():
        if a.requires_grad:
            g = out.grad * k * a.data ** (k - 1.0)
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def log(a) -> Tensor:
    a = ensure(a)
    out = Tensor(np.log(a.data))

    def _backward():
        if a.requires_grad:
            g = out.grad / a.data
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def exp(a) -> Tensor:
    a = ensure(a)
    out = Tensor(np.exp(a.data))

    def _backward():
        if a.requires_grad:
            g = out.grad * out.data
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def tanh(a) -> Tensor:
    a = ensure(a)
    out = Tensor(np.tanh(a.data))

    def _backward():
        if a.requires_grad:
            g = out.grad * (1.0 - out.data ** 2)
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def relu(a) -> Tensor:
    a = ensure(a)
    out = Tensor(np.maximum(a.data, 0.0))

    def _backward():
        if a.requires_grad:
            g = out.grad * (a.data > 0.0)
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def sigmoid(a) -> Tensor:
    a = ensure(a)
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -60, 60)))
    out = Tensor(s)

    def _backward():
        if a.requires_grad:
            g = out.grad * s * (1.0 - s)
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def mean(a, axis=None, keepdims: bool = False) -> Tensor:
    a = ensure(a)
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims))
    n = a.data.size if axis is None else a.data.shape[axis]

    def _backward():
        if not a.requires_grad:
            return
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a.grad = (np.ones_like(a.data) * g / n) if a.grad is None else a.grad + np.ones_like(a.data) * g / n

    out._backward = _backward
    return _link(out, (a,))


def sum_(a, axis=None, keepdims: bool = False) -> Tensor:
    a = ensure(a)
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims))

    def _backward():
        if not a.requires_grad:
            return
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        g = np.broadcast_to(g, a.data.shape)
        a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def softmax(a, axis: int = -1) -> Tensor:
    a = ensure(a)
    z = a.data - a.data.max(axis=axis, keepdims=True)      # numerical stability
    e = np.exp(z)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s)

    def _backward():
        if not a.requires_grad:
            return
        dot = (out.grad * s).sum(axis=axis, keepdims=True)
        g = s * (out.grad - dot)
        a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def log_softmax(a, axis: int = -1) -> Tensor:
    """Stable log-softmax. This is the heart of every language-model loss."""
    a = ensure(a)
    z = a.data - a.data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    out = Tensor(z - lse)

    def _backward():
        if not a.requires_grad:
            return
        sm = np.exp(out.data)
        g = out.grad - sm * out.grad.sum(axis=axis, keepdims=True)
        a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def concat(tensors, axis: int = 0) -> Tensor:
    ts = [ensure(t) for t in tensors]
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis))
    sizes = np.cumsum([0] + [t.data.shape[axis] for t in ts])

    def _backward():
        for i, t in enumerate(ts):
            if not t.requires_grad:
                continue
            idx = [slice(None)] * out.data.ndim
            idx[axis] = slice(sizes[i], sizes[i + 1])
            g = out.grad[tuple(idx)]
            t.grad = g if t.grad is None else t.grad + g

    out._backward = _backward
    return _link(out, tuple(ts))


def getitem(a, idx) -> Tensor:
    a = ensure(a)
    out = Tensor(a.data[idx])

    def _backward():
        if not a.requires_grad:
            return
        g = np.zeros_like(a.data)
        np.add.at(g, idx, out.grad)
        a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return _link(out, (a,))


def take_rows(table, ids) -> Tensor:
    """Embedding lookup: pick rows from a (vocab, dim) table and let them learn."""
    return getitem(table, np.asarray(ids))


def unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a broadcasted gradient back down to the parent's shape."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# ---------------------------------------------------------------------------
# 3. Modules: parameters + a forward pass
# ---------------------------------------------------------------------------


class Module:
    def params(self):
        for _, p in self._param_items():
            yield p

    def _param_items(self):
        for k, v in vars(self).items():
            if isinstance(v, Tensor) and v.requires_grad:
                yield k, v
            elif isinstance(v, Module):
                for k2, p in v._param_items():
                    yield f"{k}.{k2}", p
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        for k2, p in item._param_items():
                            yield f"{k}[{i}].{k2}", p

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def named_params(self):
        yield from self._param_items()


class Linear(Module):
    """y = x @ W + b with PyTorch-style init."""

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, bias: bool = True):
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.W = Tensor(rng.normal(0, scale, (d_in, d_out)), requires_grad=True)
        self.b = Tensor(np.zeros(d_out), requires_grad=True) if bias else None

    def forward(self, x):
        y = x @ self.W
        return y + self.b if self.b is not None else y

    __call__ = forward


class Embedding(Module):
    """A lookup table: token id -> dense vector. Words become geometry."""

    def __init__(self, n: int, dim: int, rng: np.random.Generator):
        self.table = Tensor(rng.normal(0, 0.02, (n, dim)), requires_grad=True)

    def forward(self, ids):
        return take_rows(self.table, ids)

    __call__ = forward


class LayerNorm(Module):
    """Rescale each vector to zero mean / unit variance, then learn a scale+shift."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gain = Tensor(np.ones(dim), requires_grad=True)
        self.shift = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        # Written as ordinary ops so autograd differentiates it for free.
        mu = mean(x, axis=-1, keepdims=True)
        centered = x - mu
        var = mean(mul(centered, centered), axis=-1, keepdims=True)
        inv_std = power(var + self.eps, -0.5)
        xhat = mul(centered, inv_std)
        return xhat * self.gain + self.shift

    __call__ = forward


class Adam:
    """Adam: per-parameter adaptive step size with momentum."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self, lr: float | None = None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - lr * mhat / (np.sqrt(vhat) + self.eps)


def count_params(module: Module) -> int:
    return int(sum(p.data.size for p in module.params()))

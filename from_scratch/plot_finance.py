"""Figures for the finance experiment.

Two pictures, both of which are more informative than a Sharpe ratio table:

  finance_curves.png - what the money actually did, out of sample, including
                       the buy-and-hold benchmark and the oracle ceiling.
  finance_tail.png   - the honest bit: on simulated data the models look
                       amazing, but the same universe with the edge removed
                       shows all of them falling back to zero. That comparison
                       is the whole point.

Run:  python plot_finance.py            (after finance.py has written its JSON)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

COLORS = {
    "zero (coin flip)": "#9aa0a6",
    "ridge (linear)": "#4285f4",
    "mlp (from scratch)": "#ea4335",
    "rnn (from scratch)": "#fbbc04",
    "ORACLE (cheating)": "#34a853",
}


def load(mode):
    with open(os.path.join(RESULTS, f"finance_{mode}.json")) as f:
        return json.load(f)


def curves(runs, title, path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    for key, col in COLORS.items():
        if key not in runs:
            continue
        style = dict(color=col, lw=2.0, label=key)
        if key == "ORACLE (cheating)":
            style.update(lw=1.4, ls="--")
        ax.plot(runs[key], **style)
    ax.axhline(1_000_000, color="#202124", lw=0.8, alpha=0.4, ls=":")
    ax.set_xlabel("trading day (out of sample)")
    ax.set_ylabel("equity, starting at $1,000,000")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1e6:.0f}M" if v < 1e7 else f"${v / 1e6:.0f}M")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _mean(v):
    return v["mean"] if isinstance(v, dict) else v[0]


def _std(v):
    return v["std"] if isinstance(v, dict) else v[1]


def tail_figure(signal_agg, noise_agg, path):
    """OLS R^2 of each model against the noise floor, side by side."""
    models = [m for m in signal_agg if m != "ORACLE (cheating)"]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=130)
    x = np.arange(len(models))
    w = 0.36
    for offset, agg, name, col in [(-w / 2, signal_agg, "world with an edge", "#1a73e8"),
                                   (w / 2, noise_agg, "same world, edge removed", "#9aa0a6")]:
        vals = [_mean(agg[m]["r2"]) for m in models]
        errs = [_std(agg[m]["r2"]) for m in models]
        ax.bar(x + offset, vals, w, yerr=errs, capsize=3, label=name, color=col, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("out-of-sample $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("A model only earns its keep where an edge exists")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make(everything):
    """everything: {mode: dict(runs=[...], agg=..., bh=..., ceiling=...)}"""
    if "signal" in everything:
        runs = everything["signal"]["runs"][0]
        eq = {name: row["equity"] for name, row in runs["rows"].items()}
        curves(eq, "Out-of-sample equity, 20% long / 20% short, 2 bps costs",
               os.path.join(RESULTS, "finance_curves.png"))

    if "signal" in everything and "noise" in everything:
        tail_figure(everything["signal"]["agg"], everything["noise"]["agg"],
                    os.path.join(RESULTS, "finance_tail.png"))


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "finance_signal.json")) as f:
        sig = json.load(f)
    with open(os.path.join(RESULTS, "finance_noise.json")) as f:
        noi = json.load(f)
    tail_figure(sig["models"], noi["models"], os.path.join(RESULTS, "finance_tail.png"))
    print("wrote", os.path.join(RESULTS, "finance_tail.png"))

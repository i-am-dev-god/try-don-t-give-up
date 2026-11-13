"""A miniature "stock market analyser", built end to end from scratch.

What this does
--------------
1. Simulates a market we fully understand, so success can be measured honestly.
   Prices have: regime switching, volatility clustering, a fat-tailed shock
   process, a market factor -- and one small, real effect: when a stock's
   recent volume has surged (attention is high), part of its 5-day move
   reverses tomorrow. On quiet days it does not. The effect is an interaction,
   so a purely linear model can only learn its average, which is wrong in
   both regimes.

2. Builds the features a desk would compute: lagged returns, rolling vol,
   momentum, volume surprise, attention vs baseline, drawdown, market context.

3. Trains four predictors on a strict chronological split (70/15/15, never
   shuffled across time):
      zero   - "tomorrow is a coin flip": the efficient-market baseline
      ridge  - linear regression on the flattened window, closed form
      mlp    - the neural net in models.py, on summarised features
      rnn    - the recurrent net in models.py, on the raw window

4. Evaluates the only ways that matter: out-of-sample R^2 against the
   theoretical ceiling, rank IC, directional hit rate, and a cost-aware
   long/short backtest. Everything is averaged over several seeds.

Run:
    python finance.py                # both worlds, 3 seeds (~4 min, 2 CPU cores)
    python finance.py --seeds 1      # faster
    python finance.py --quick        # smoke test, a couple of seconds

The second world is the important control: the *same* market with the edge
switched off. A real edge shows up only in the first. If a model looks good
in both, it is fitting noise.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from models import MLP, RNN
from nn import Adam, Tensor, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. The world
# ---------------------------------------------------------------------------


def simulate_market(n_stocks: int = 40, n_days: int = 2000, seed: int = 7,
                    edge: bool = True):
    """Return log-returns (days x stocks), volumes, and the true conditional mean.

    The generative story, in order:
      regime   - a hidden Markov chain flips between calm / normal / crisis.
                 Drift and market volatility change with it. Nobody observes
                 the regime; you only see its footprints.
      vol      - market volatility follows GARCH(1,1): big days cluster.
      factor   - every stock loads on one common market return with its beta.
      reversal - on days when a stock's *5-day* volume has surged (attention
                 is high, other people are in the trade) part of the last
                 5-day move bleeds back out tomorrow. On quiet days it does
                 not. The effect is an interaction: (move) x (attention), so a
                 linear model can only recover its average and is wrong in both
                 regimes. That is where a nonlinear model earns its keep.

    Everything is log-returns, so "return" here means log-return throughout.
    """
    rng = np.random.default_rng(seed)

    # --- hidden volatility regime -------------------------------------
    states = np.array([0, 1, 2])            # calm, normal, crisis
    drift = np.array([0.0006, 0.0002, -0.0012])
    vol_mult = np.array([0.6, 1.0, 2.2])
    transition = np.array([[0.985, 0.014, 0.001],
                           [0.010, 0.975, 0.015],
                           [0.005, 0.095, 0.900]])
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = rng.choice(states, p=transition[regime[t - 1]])

    # --- GARCH(1,1) market volatility ---------------------------------
    base_vol = 0.009
    omega, alpha, beta = 0.00002, 0.10, 0.86
    market = np.zeros(n_days)
    var = np.zeros(n_days)
    for t in range(1, n_days):
        long_run = (base_vol * vol_mult[regime[t]]) ** 2
        var[t] = omega + alpha * market[t - 1] ** 2 + beta * var[t - 1]
        var[t] = 0.25 * var[t] + 0.75 * long_run      # keep near the regime level
        sigma = np.sqrt(var[t])
        shock = rng.standard_t(df=4) / np.sqrt(2.0)   # fat tails, unit variance
        market[t] = drift[regime[t]] + sigma * shock

    # --- the cross-section --------------------------------------------
    betas = rng.uniform(0.6, 1.5, n_stocks)
    idio = rng.uniform(0.008, 0.020, n_stocks)
    common = rng.standard_t(df=5, size=n_days) / np.sqrt(5.0 / 3.0)
    log_volume = np.zeros((n_days, n_stocks))
    stock_ret = np.zeros((n_days, n_stocks))
    true_mean = np.zeros((n_days, n_stocks))          # E[r_t | info available at t-1]

    for t in range(1, n_days):
        eps = rng.standard_t(df=5, size=n_stocks) / np.sqrt(5.0 / 3.0)
        stock_ret[t] = betas * market[t] + 0.4 * idio * common[t] + idio * eps

        # volume this day: baseline chatter + the day's own move + a
        # persistent "this stock is in the news" state that decays slowly
        log_volume[t] = (1.0 + 0.6 * log_volume[t - 1]
                         + 4.0 * np.abs(stock_ret[t])
                         + rng.normal(0, 0.35, n_stocks))

        if t >= 60:
            move5 = stock_ret[t - 5:t].sum(axis=0)
            # attention: 5-day volume vs this stock's own 60-day baseline,
            # measured in log units, so +0.35 means "about 40% above normal"
            attn5 = log_volume[t - 5:t].mean(axis=0)
            base = log_volume[t - 60:t].mean(axis=0)
            surge = (attn5 - base) > 0.35
            pull = -0.45 * move5 * surge
            if edge:
                stock_ret[t] += pull
                true_mean[t] = pull

    return dict(ret=stock_ret, volume=np.exp(log_volume), market=market,
                regime=regime, true_mean=true_mean, var=var, betas=betas)


# ---------------------------------------------------------------------------
# 2. Features: what a desk would actually compute
# ---------------------------------------------------------------------------

WINDOW = 10          # days of history fed to the model
N_FEATURES = 15
MLP_IN = 3 * N_FEATURES      # summarise() packs last / mean / std


def rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Rolling window stat over axis 0, aligned so row t uses data <= t."""
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(w - 1, x.shape[0]):
        out[t] = fn(x[t - w + 1:t + 1], axis=0)
    return out


def build_dataset(m: dict, window: int = WINDOW):
    """Return X (samples, window, features), y (samples,), dates (samples,)."""
    r, v, mkt = m["ret"], m["volume"], m["market"]
    n_days, n_stocks = r.shape

    vol20 = rolling(r, 20, np.std)
    vol5 = rolling(r, 5, np.std)
    mom5 = rolling(r, 5, np.sum)
    mom20 = rolling(r, 20, np.sum)
    vol_ratio = vol5 / (vol20 + 1e-9)
    volz = (np.log(v) - rolling(np.log(v), 20, np.mean)) / (rolling(np.log(v), 20, np.std) + 1e-9)
    z = r / (vol20 + 1e-9)
    mkt5 = np.array([mkt[max(0, t - 4):t + 1].sum() for t in range(n_days)])
    mkt_vol = np.array([np.std(mkt[max(0, t - 19):t + 1]) for t in range(n_days)])
    vol_shift = vol20 / (np.roll(vol20, 5, axis=0) + 1e-9)
    # the same "is this stock in the news" gate the world uses: 5-day average
    # volume vs the long-run level, expressed in log units
    attn5 = rolling(np.log(v + 1e-9), 5, np.mean)
    attn_base = rolling(np.log(v + 1e-9), 60, np.mean)
    attention = attn5 - attn_base
    # drawdown from the 20-day running high of cumulative return
    cum = np.cumsum(r, axis=0)
    run_max = np.maximum.accumulate(np.stack([np.roll(cum, i, axis=0) for i in range(20)]), axis=0)[19]
    drawdown = cum - run_max

    feats = np.stack([
        r,                                       # today's return
        np.roll(r, 1, axis=0),                   # yesterday's
        vol20,                                   # realised vol
        vol_ratio,                               # short vs long vol
        mom5,                                    # 5-day momentum
        mom20,                                   # 20-day momentum
        volz,                                    # volume surprise
        attention,                               # 5-day attention vs baseline
        np.abs(z),                               # standardised move size
        np.broadcast_to(mkt5[:, None], r.shape),                 # market momentum
        np.broadcast_to(mkt_vol[:, None], r.shape),              # market vol
        vol_shift,                               # volatility regime shift
        drawdown,                                # distance from 20d high
        (r > 0).astype(float),                   # up/down indicator
        np.broadcast_to(np.arange(n_days)[:, None] / n_days, r.shape),  # time trend
    ], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.shape[-1] == N_FEATURES, feats.shape

    # Contract: a sample's `date` is the day the *label* lands on, so sample d
    # predicts return[d]. Features therefore stop at d-1 -- no lookahead.
    X, y, dates = [], [], []
    for t in range(window, n_days - 1):
        X.append(feats[t - window:t])                    # history strictly before t
        y.append(r[t])                                   # return on day t
        dates.append(np.full(n_stocks, t))
    X = np.concatenate(X, axis=1)                        # (window, days*stocks, feat)
    y = np.concatenate(y, axis=0)
    dates = np.concatenate(dates, axis=0)
    X = np.transpose(X, (1, 0, 2))
    return X, y, dates


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------


def summarise(X):
    """Per-stock summary of a window: last day, mean, std -> 3 * N_FEATURES.

    Feeding a 10-day window to a dense net is mostly noise: 150 inputs for a
    signal that lives in a handful of them. This is the feature engineering
    step every real desk does, and it is fair to give it to every model.
    """
    return np.concatenate([X[:, -1, :], X.mean(axis=1), X.std(axis=1)], axis=1)


class Zero:
    name = "zero (coin flip)"

    def fit(self, *a, **k):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Ridge:
    """Linear regression with an L2 penalty, solved in closed form."""

    name = "ridge (linear)"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xf = X.reshape(len(X), -1)
        A = Xf.T @ Xf + self.alpha * np.eye(Xf.shape[1])
        self.w = np.linalg.solve(A, Xf.T @ y)
        return self

    def predict(self, X):
        return X.reshape(len(X), -1) @ self.w


def _sq_err(pred, y):
    d = pred - y
    return (d * d).mean()


def train_torch_like(model, Xtr, ytr, Xva, yva, *, epochs=6, batch=256, lr=2e-3,
                     clip=1.0, log_every=0, seed=0, name="", weight_decay=0.0):
    """A standard training loop: shuffle, forward, loss, backward, step."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    best = (np.inf, None)
    history = []
    n = len(Xtr)
    for ep in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb = Tensor(Xtr[idx])
            logits = model(xb)
            pred = logits[:, 0] if logits.shape[-1] == 1 else logits
            diff = pred - ytr[idx]
            loss = (diff * diff).mean()
            model.zero_grad()
            loss.backward()
            for p in model.params():                    # clipping + weight decay
                if p.grad is not None:
                    np.clip(p.grad, -clip, clip, out=p.grad)
                    if weight_decay:
                        p.grad = p.grad + 2.0 * weight_decay * p.data
            opt.step()
            total += float(loss.data) * len(idx)
        train_loss = total / n
        va = float(_sq_err(model(Tensor(Xva))[:, 0], yva)) if Xva is not None else np.nan
        history.append((train_loss, va))
        if va < best[0]:
            best = (va, [(n_, p.data.copy()) for n_, p in model.named_params()])
        if log_every and ep % log_every == 0:
            print(f"    {name} epoch {ep:2d}  train mse {train_loss:.3e}  val mse {va:.3e}")
    if best[1] is not None:                              # restore best-on-val
        for (nm, val), p in zip(best[1], model.params()):
            p.data = val.copy()
    return model, history


# ---------------------------------------------------------------------------
# 4. Evaluation: does the thing actually make money?
# ---------------------------------------------------------------------------


def r2_score(y, p) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def directional_accuracy(y, p) -> float:
    """Hit rate on the days the model actually takes a side.

    A flat prediction has no direction, so scoring it as a miss would punish
    a model for being honest about having no view -- count only |p| > 0.
    """
    active = np.abs(p) > 1e-12
    if active.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y[active]) == np.sign(p[active])))


def cross_sectional_ic(y, p, dates) -> float:
    """Spearman-style rank correlation, computed per day then averaged."""
    ics = []
    for d in np.unique(dates):
        sel = dates == d
        if sel.sum() < 5:
            continue
        ry = np.argsort(np.argsort(y[sel]))
        rp = np.argsort(np.argsort(p[sel]))
        ics.append(np.corrcoef(ry, rp)[0, 1])
    return float(np.nanmean(ics))


def backtest(y, p, dates, top_frac=0.2, cost_bps=2.0, capital=1_000_000):
    """Long the top quintile of predictions, short the bottom quintile.

    Returns the daily return series after trading costs, plus stats.
    """
    prev_w = {}
    daily = []
    turnover = []
    for d in np.unique(dates).tolist():
        sel = dates == d
        pv, rv = p[sel], y[sel]
        k = max(1, int(len(pv) * top_frac))
        order = np.argsort(pv)
        w = np.zeros(len(pv))
        w[order[-k:]] = 1.0 / k
        w[order[:k]] = -1.0 / k
        gross = float(w @ rv)
        prev = prev_w.get(d - 1, np.zeros(len(pv)))
        turn = 0.5 * float(np.abs(w - prev).sum())
        cost = turn * cost_bps / 1e4
        daily.append(gross - cost)
        turnover.append(turn)
        prev_w[d] = w
    daily = np.array(daily)
    equity = capital * np.exp(np.cumsum(daily))
    ann = 252.0
    sharpe = float(np.mean(daily) / (np.std(daily) + 1e-12) * np.sqrt(ann))
    total = float(equity[-1] / capital - 1.0)
    peak = np.maximum.accumulate(equity)
    dd = float(((equity - peak) / peak).min())
    return dict(daily=daily, equity=equity, sharpe=sharpe, total_return=total,
                max_drawdown=dd, avg_turnover=float(np.mean(turnover)))


def buy_and_hold(market_returns):
    eq = np.exp(np.cumsum(market_returns))
    sharpe = float(np.mean(market_returns) / (np.std(market_returns) + 1e-12) * np.sqrt(252))
    return eq, sharpe


# ---------------------------------------------------------------------------
# 5. One full run of the experiment
# ---------------------------------------------------------------------------


def run_once(mode: str, stocks: int, days: int, epochs: int, seed: int,
             verbose: bool = True, mlp_epochs: int = 25, mlp_lr: float = 1e-3,
             mlp_ensemble: int = 3):
    """Simulate a world, train every model, evaluate once. Returns a dict."""
    m = simulate_market(stocks, days, seed=seed, edge=(mode == "signal"))
    X, y, dates = build_dataset(m)

    # --- strict chronological split: 70 / 15 / 15 ---------------------
    all_days = np.unique(dates)
    t_tr = all_days[int(0.70 * len(all_days))]
    t_va = all_days[int(0.85 * len(all_days))]
    tr = dates < t_tr
    va = (dates >= t_tr) & (dates < t_va)
    te = dates >= t_va

    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-9
    Xs = np.clip((X - mu) / sd, -8, 8)

    y_mu, y_sd = y[tr].mean(), y[tr].std()      # forecast in standardised units
    ytr = (y[tr] - y_mu) / y_sd
    yva = (y[va] - y_mu) / y_sd
    y_raw = y[te]                              # test labels in raw return units

    rng = np.random.default_rng(seed + 1)
    preds, params = {}, {}

    preds["zero (coin flip)"] = np.zeros(te.sum())
    params["zero (coin flip)"] = 0

    # --- linear model: ridge on the flattened window -------------------
    ridge = Ridge(alpha=10.0).fit(Xs[tr], ytr)
    preds["ridge (linear)"] = ridge.predict(Xs[te])
    params["ridge (linear)"] = Xs.shape[-1] * WINDOW

    # --- neural nets: MLP on summarised features, RNN on the raw window -
    # Each gets an input representation that is reasonable for its shape.
    # That is what a practitioner would do; comparing a well-tuned linear
    # model against a badly-fed neural net proves nothing.
    Str, Sva, Ste = summarise(Xs[tr]), summarise(Xs[va]), summarise(Xs[te])

    mlp_ens = []
    for k in range(mlp_ensemble):
        net = MLP(MLP_IN, 32, 1, np.random.default_rng(seed + 10 + k), depth=2)
        net, hist = train_torch_like(net, Str, ytr, Sva, yva, epochs=mlp_epochs,
                                     batch=256, lr=mlp_lr, weight_decay=1e-5)
        mlp_ens.append(net(Tensor(Ste)).data[:, 0])
    preds["mlp (from scratch)"] = np.mean(mlp_ens, axis=0)
    params["mlp (from scratch)"] = count_params(net) * mlp_ensemble

    rnn_ens = []
    for k in range(mlp_ensemble):
        rec = RNN(N_FEATURES, 32, 1, np.random.default_rng(seed + 20 + k))
        rec, _ = train_torch_like(rec, Xs[tr], ytr, Xs[va], yva, epochs=max(2, mlp_epochs // 2),
                                  batch=256, lr=mlp_lr, weight_decay=1e-5)
        rnn_ens.append(rec(Tensor(Xs[te])).data[:, 0])
    preds["rnn (from scratch)"] = np.mean(rnn_ens, axis=0)
    params["rnn (from scratch)"] = count_params(rec) * mlp_ensemble

    # --- the ceiling: what a perfect forecaster could score ------------
    # Only the deterministic part of a return is predictable; the rest is
    # coin flip and no amount of training removes it. So the best possible
    # predictor is E[r | info], which the oracle below knows exactly.
    cols = np.tile(np.arange(stocks), len(all_days))
    tm_all = m["true_mean"][dates, cols]
    noise_var = float(np.var(y - tm_all))
    ceiling = float(np.var(tm_all) / (np.var(tm_all) + noise_var))

    def evaluate(pred_ret):
        bt = backtest(y_raw, pred_ret, dates[te])
        return dict(r2=r2_score(y_raw, pred_ret),
                    hit=directional_accuracy(y_raw, pred_ret),
                    coverage=float(np.mean(np.abs(pred_ret) > 1e-12)),
                    ic=cross_sectional_ic(y_raw, pred_ret, dates[te]),
                    sharpe=bt["sharpe"], ret=bt["total_return"],
                    mdd=bt["max_drawdown"], turnover=bt["avg_turnover"],
                    equity=bt["equity"])

    rows = {}
    for name, p in preds.items():
        stats = evaluate(p * y_sd + y_mu)      # de-standardise before scoring
        stats["params"] = params[name]
        stats["pred"] = p * y_sd + y_mu
        rows[name] = stats
    rows["ORACLE (cheating)"] = dict(evaluate(tm_all[te]), params=0,
                                     pred=tm_all[te], cheat=True)

    # buy & hold: the equal-weight book over the same test days
    mkt_test = m["ret"].mean(axis=1)[all_days[all_days >= t_va]]
    bh_eq, bh_sharpe = buy_and_hold(mkt_test)
    bh = dict(sharpe=bh_sharpe, ret=float(bh_eq[-1] - 1), equity=bh_eq)

    out = dict(mode=mode, seed=seed, ceiling=ceiling, rows=rows, bh=bh,
               n_test=int(te.sum()), dates=dates[te], y_test=y_raw,
               oracle_pred=tm_all[te])
    if verbose:
        print(f"   seed {seed}: {X.shape[0]:,} stock-days "
              f"({X.shape[1]}d context x {X.shape[2]} features), "
              f"train {tr.sum():,} / val {va.sum():,} / test {te.sum():,}")
        print(f"   R^2 ceiling for this world: {ceiling:+.4f}")
        print(f"   {'model':22s} {'params':>8s} {'R2(oos)':>9s} {'hit':>7s} "
              f"{'IC':>7s} {'Sharpe':>7s} {'ret':>8s} {'maxDD':>7s} {'turn':>6s}")
        for name, r in rows.items():
            pstr = f"{r['params']:,d}" if r["params"] else "-"
            print(f"   {name:22s} {pstr:>8s} {r['r2']:+9.4f} {r['hit']:7.2%} "
                  f"{r['ic']:+7.3f} {r['sharpe']:7.2f} {r['ret']:+8.1%} "
                  f"{r['mdd']:7.1%} {r['turnover']:6.2f}")
        print(f"   {'buy & hold market':22s} {'-':>8s} {'-':>9s} {'-':>7s} "
              f"{'-':>7s} {bh_sharpe:7.2f} {bh_eq[-1] - 1:8.1%} {'-':>7s} {'-':>6s}")
    return out


def aggregate(runs):
    """Mean +/- std across seeds for every model and every metric."""
    names = list(runs[0]["rows"].keys())
    agg = {}
    for name in names:
        metrics = {}
        for key in ("r2", "hit", "ic", "sharpe", "ret", "mdd", "turnover"):
            vals = np.array([r["rows"][name][key] for r in runs])
            metrics[key] = (float(np.mean(vals)), float(np.std(vals)))
        metrics["params"] = runs[0]["rows"][name]["params"]
        agg[name] = metrics
    bh = {}
    for key in ("sharpe", "ret"):
        vals = np.array([r["bh"][key] for r in runs])
        bh[key] = (float(np.mean(vals)), float(np.std(vals)))
    ceiling = (float(np.mean([r["ceiling"] for r in runs])),
               float(np.std([r["ceiling"] for r in runs])))
    return agg, bh, ceiling


def print_aggregate(agg, bh, ceiling, n_seeds):
    print(f"\n   === {n_seeds} seeds, mean +/- std ===")
    print(f"   R^2 ceiling: {ceiling[0]:+.4f} +/- {ceiling[1]:.4f}")
    print(f"   {'model':22s} {'params':>8s} {'R2(oos)':>17s} {'hit':>14s} "
          f"{'IC':>14s} {'Sharpe':>16s}")
    for name, m in agg.items():
        pstr = f"{m['params']:,d}" if m["params"] else "-"
        print(f"   {name:22s} {pstr:>8s} {m['r2'][0]:+8.4f}±{m['r2'][1]:6.4f} "
              f"{m['hit'][0]:7.2%}±{m['hit'][1]:5.2%} "
              f"{m['ic'][0]:+7.3f}±{m['ic'][1]:5.3f} "
              f"{m['sharpe'][0]:7.2f}±{m['sharpe'][1]:5.2f}")
    print(f"   {'buy & hold market':22s} {'-':>8s} {'-':>17s} {'-':>14s} {'-':>14s} "
          f"{bh['sharpe'][0]:7.2f}±{bh['sharpe'][1]:5.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["signal", "noise", "both"], default="both")
    ap.add_argument("--stocks", type=int, default=40)
    ap.add_argument("--days", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="small and fast, for CI")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.seeds = 1, 1
        args.stocks, args.days = max(10, args.stocks // 2), 600

    modes = ["signal", "noise"] if args.mode == "both" else [args.mode]
    t0 = time.time()
    everything = {}
    for mode in modes:
        print(f"\n== world: {'real edge' if mode == 'signal' else 'pure noise'} "
              f"({args.stocks} stocks x {args.days} days) ==")
        runs = []
        for i in range(args.seeds):
            runs.append(run_once(mode, args.stocks, args.days, args.epochs,
                                 seed=7 + 13 * i))
        agg, bh, ceiling = aggregate(runs)
        print_aggregate(agg, bh, ceiling, args.seeds)
        everything[mode] = dict(runs=runs, agg=agg, bh=bh, ceiling=ceiling)

    # --- persist ------------------------------------------------------
    for mode, blob in everything.items():
        summary = dict(
            mode=mode, seeds=args.seeds, stocks=args.stocks, days=args.days,
            epochs=args.epochs,
            ceiling=dict(mean=blob["ceiling"][0], std=blob["ceiling"][1]),
            models={k: {mk: dict(mean=mv[0], std=mv[1]) if isinstance(mv, tuple) else mv
                        for mk, mv in m.items()}
                    for k, m in blob["agg"].items()},
            buy_hold=dict(sharpe=dict(mean=blob["bh"]["sharpe"][0], std=blob["bh"]["sharpe"][1]),
                          ret=dict(mean=blob["bh"]["ret"][0], std=blob["bh"]["ret"][1])),
        )
        with open(os.path.join(RESULTS, f"finance_{mode}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    try:
        import plot_finance
        plot_finance.make(everything)
        print(f"\n   wrote {os.path.join(RESULTS, 'finance_*.png')}")
    except ImportError:
        pass

    print(f"\n   done in {time.time() - t0:.0f}s")
    return everything


if __name__ == "__main__":
    main()

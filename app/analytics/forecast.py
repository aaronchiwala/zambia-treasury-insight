"""Forecasting with honest uncertainty.

Three direct multi-horizon models, all producing *distributions* (5/25/75/95% quantiles), not just a point:

  rw      random walk: centre = last level; band = empirical spread of past h-step changes.
  driver  ridge regression of the h-step-ahead log change on the last k periods of driver moves
          (copper, dollar, oil, rates, own momentum). Uses only information known at the forecast
          origin, so no assumptions about future driver values are needed. Drift-free by design.
  blend   50/50 mix of the two (usually the most robust when drivers have modest signal).

Every model is judged by rolling-origin backtests (see backtest.py) against the random walk, and the UI
shows that track record next to every forecast. If a model has no demonstrated edge, the output says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import GROUP_LABELS, TARGET
from .attribution import _select_common_sample
from .panel import Panel

QUANTILES = (0.05, 0.25, 0.75, 0.95)
MODELS = ("rw", "driver", "blend")


@dataclass
class Pred:
    mid: float          # log change, median forecast
    q: dict[float, float]  # quantile -> log change
    contrib: dict[str, float] | None = None  # feature contributions (log pts), driver/blend only


class ForecastContext:
    def __init__(self, panel: Panel, k: int = 4, min_obs: int = 104):
        self.panel = panel
        rets = panel.returns
        _, keep, self.excluded = _select_common_sample(rets, panel.drivers, min_obs)
        self.drivers_used = keep
        cols = [TARGET] + keep
        feats = rets[cols].rolling(k, min_periods=k).sum()
        feats.columns = [("momentum" if c == TARGET else c) for c in cols]
        self.X = feats
        self.k = k
        lv = panel.levels[TARGET].reindex(rets.index)
        self.ylog = np.log(lv.values.astype(float))
        self.index = rets.index
        self.n = len(rets)
        self.Xv = feats.values.astype(float)
        self.kinds = {c: ("price" if c == "momentum" else panel.kinds[c]) for c in feats.columns}

    # ---- single model fits -------------------------------------------------
    def _train_slice(self, t: int, h: int, window: int):
        lo = max(0, t - h - window)
        s = np.arange(lo, t - h + 1)           # origins whose h-step outcome is known by time t
        s = s[s + h <= t]
        return s

    def predict(self, t: int, h: int, window: int = 260, ridge: float = 10.0) -> dict[str, Pred]:
        s = self._train_slice(t, h, window)
        if len(s) < 40:
            raise ValueError("Not enough history to forecast at this origin.")
        tgt = self.ylog[s + h] - self.ylog[s]
        good = ~np.isnan(tgt)
        s, tgt = s[good], tgt[good]
        # ---- random walk
        c = tgt - np.median(tgt)
        rw = Pred(0.0, {q: float(np.quantile(c, q)) for q in QUANTILES})
        # ---- driver model
        Xtr = self.Xv[s]
        ok = ~np.isnan(Xtr).any(axis=1)
        x_t = self.Xv[t]
        if ok.sum() < 40 or np.isnan(x_t).any():
            return {"rw": rw, "driver": rw, "blend": rw}
        Xtr, ytr = Xtr[ok], tgt[ok]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        Z = (Xtr - mu) / sd
        p = Z.shape[1]
        beta = np.linalg.solve(Z.T @ Z + ridge * np.eye(p), Z.T @ ytr)   # no intercept: drift-free
        resid = ytr - Z @ beta
        infl = np.sqrt(len(ytr) / max(len(ytr) - p, 1))
        r_c = (resid - np.median(resid)) * infl
        z_t = (x_t - mu) / sd
        contrib = dict(zip(self.X.columns, (beta * z_t).tolist()))
        mid = float(z_t @ beta + np.median(resid))
        drv = Pred(mid, {q: float(np.quantile(r_c, q)) for q in QUANTILES}, contrib)
        # ---- blend: average centres and band half-widths
        bl = Pred(0.5 * mid, {q: 0.5 * (rw.q[q] + drv.q[q]) for q in QUANTILES},
                  {k_: 0.5 * v for k_, v in contrib.items()})
        return {"rw": rw, "driver": drv, "blend": bl}


def to_levels(last_level: float, p: Pred) -> dict:
    f = lambda x: float(last_level * np.exp(x))
    return dict(mid=f(p.mid), q05=f(p.mid + p.q[0.05]), q25=f(p.mid + p.q[0.25]),
                q75=f(p.mid + p.q[0.75]), q95=f(p.mid + p.q[0.95]), change_pct=float((np.exp(p.mid) - 1) * 100))


def select_model(summary: list[dict] | None) -> tuple[str, str]:
    """Pick the model with the best mean backtest MAE; fall back to the random walk if nothing beats it."""
    if not summary:
        return "blend", "No backtest available; defaulting to the blend."
    by = {}
    for r in summary:
        by.setdefault(r["model"], []).append(r["mae_pct"])
    avg = {m: float(np.mean(v)) for m, v in by.items()}
    best = min(avg, key=avg.get)
    rw = avg.get("rw", np.inf)
    if best != "rw" and avg[best] < 0.99 * rw:
        return best, f"'{best}' had the lowest average backtest error ({avg[best]:.2f}% vs {rw:.2f}% for the random walk)."
    return "rw", "No model beat the random walk in backtests, so the centre is held flat and only the uncertainty band is informative."


def make_forecast(panel: Panel, h_max: int = 12, model: str = "auto", summary: list[dict] | None = None) -> dict:
    ctx = ForecastContext(panel)
    t = ctx.n - 1
    last = float(panel.levels[TARGET].iloc[-1])
    last_date = panel.levels.index[-1]
    if model == "auto":
        chosen, why = select_model(summary)
    elif model in MODELS:
        chosen, why = model, "Selected manually."
    else:
        raise ValueError(f"Unknown model {model!r}")
    dates = pd.date_range(last_date, periods=h_max + 1, freq=panel.freq)[1:]
    path, contrib_last = [], None
    for h in range(1, h_max + 1):
        preds = ctx.predict(t, h)
        lv = to_levels(last, preds[chosen])
        lv.update(h=h, date=dates[h - 1].strftime("%Y-%m-%d"))
        path.append(lv)
        if h == h_max:
            contrib_last = preds["driver"].contrib  # always show what the driver model leans on
    drivers_view = []
    if contrib_last:
        for name, v in sorted(contrib_last.items(), key=lambda kv: -abs(kv[1])):
            raw = float(ctx.X[name].iloc[-1])
            drivers_view.append(dict(
                feature=name, label="Own momentum" if name == "momentum" else GROUP_LABELS.get(name, name),
                recent_move=raw * 100 if ctx.kinds[name] == "price" else raw,
                unit="%" if ctx.kinds[name] == "price" else "pp",
                pull_pct=float((np.exp(v) - 1) * 100)))
    return dict(model=chosen, model_reason=why, last_date=last_date.strftime("%Y-%m-%d"), last_level=last,
                horizon=h_max, freq=panel.freq, path=path, drivers=drivers_view,
                drivers_used=ctx.drivers_used, excluded=ctx.excluded, lookback_periods=ctx.k)

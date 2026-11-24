"""Forecasting with calibrated 50%, 95%, and 99% confidence intervals.

Three direct multi-horizon models producing empirical return distributions:
  rw      random walk: centre = last level; bands calibrated from past h-step changes with finite-sample Student-t tail guard.
  driver  ridge regression of the h-step-ahead log change on trailing k-period driver moves
          (copper, USD, SA Rand, SA financial markets, EM sovereign debt, rates, own momentum).
  blend   50/50 mix of the two.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import GROUP_LABELS, TARGET
from .attribution import _select_common_sample
from .panel import Panel

QUANTILES = (0.005, 0.025, 0.05, 0.25, 0.75, 0.95, 0.975, 0.995)
MODELS = ("rw", "driver", "blend")

_Z_TAIL = {
    0.005: -2.575829,
    0.025: -1.959964,
    0.05: -1.644854,
    0.95: 1.644854,
    0.975: 1.959964,
    0.995: 2.575829,
}


def _t_crit(q: float, df: int) -> float:
    z = _Z_TAIL[q]
    z2 = z * z
    g1 = (z * (z2 + 1.0)) / (4.0 * df)
    g2 = (z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0)) / (96.0 * df * df)
    return float(z + g1 + g2)


def _calibrated_quantiles(samples: np.ndarray, h: int) -> dict[float, float]:
    sd = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    df_eff = max(7, int(len(samples) / max(1.0, h * 0.65)))
    out: dict[float, float] = {}
    for q in QUANTILES:
        emp = float(np.quantile(samples, q))
        if q in _Z_TAIL:
            t_bound = _t_crit(q, df_eff) * sd
            out[q] = min(emp, t_bound) if q < 0.5 else max(emp, t_bound)
        else:
            out[q] = emp
    return out


@dataclass
class Pred:
    mid: float  # log change, median forecast
    q: dict[float, float]  # quantile -> log change
    contrib: dict[str, float] | None = None


class ForecastContext:
    def __init__(self, panel: Panel, k: int = 4, min_obs: int | None = None):
        if min_obs is None:
            min_obs = 104 if panel.freq == "W-FRI" else 48
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

    def _train_slice(self, t: int, h: int, window: int):
        lo = max(0, t - h - window)
        s = np.arange(lo, t - h + 1)
        s = s[s + h <= t]
        return s

    def predict(self, t: int, h: int, window: int = 260, ridge: float = 10.0) -> dict[str, Pred]:
        s = self._train_slice(t, h, window)
        if len(s) < 40:
            raise ValueError("Not enough history to forecast at this origin.")
        tgt = self.ylog[s + h] - self.ylog[s]
        good = ~np.isnan(tgt)
        s, tgt = s[good], tgt[good]

        # Random walk with finite-sample Student-t tail guard
        rw_infl = float(np.sqrt((len(tgt) + 1) / max(len(tgt) - 1, 1)))
        centered = (tgt - np.median(tgt)) * rw_infl
        rw = Pred(0.0, _calibrated_quantiles(centered, h))

        # Driver Ridge model (proportional shrinkage with p)
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
        eff_ridge = max(ridge, p * 4.0)
        beta = np.linalg.solve(Z.T @ Z + eff_ridge * np.eye(p), Z.T @ ytr)
        resid = ytr - Z @ beta
        infl = float(np.sqrt((len(ytr) + 1) / max(len(ytr) - p, 1)))
        r_c = (resid - np.median(resid)) * infl
        z_t = (x_t - mu) / sd
        contrib = dict(zip(self.X.columns, (beta * z_t).tolist()))
        mid = float(z_t @ beta + np.median(resid))
        drv = Pred(mid, _calibrated_quantiles(r_c, h), contrib)

        # 50/50 Blend
        bl = Pred(
            0.5 * mid,
            {q: 0.5 * (rw.q[q] + drv.q[q]) for q in QUANTILES},
            {k_: 0.5 * v for k_, v in contrib.items()},
        )
        return {"rw": rw, "driver": drv, "blend": bl}


def to_levels(last_level: float, p: Pred) -> dict:
    f = lambda x: float(last_level * np.exp(x))
    return dict(
        mid=f(p.mid),
        q005=f(p.mid + p.q[0.005]),
        q025=f(p.mid + p.q[0.025]),
        q05=f(p.mid + p.q[0.05]),
        q25=f(p.mid + p.q[0.25]),
        q75=f(p.mid + p.q[0.75]),
        q95=f(p.mid + p.q[0.95]),
        q975=f(p.mid + p.q[0.975]),
        q995=f(p.mid + p.q[0.995]),
        change_pct=float((np.exp(p.mid) - 1) * 100),
    )


def select_model(summary: list[dict] | None) -> tuple[str, str]:
    """Pick the model with the best mean backtest MAE; fall back to random walk if nothing beats it."""
    if not summary:
        return "blend", "No backtest available; defaulting to the blend."
    by: dict[str, list[float]] = {}
    for r in summary:
        by.setdefault(r["model"], []).append(r["mae_pct"])
    avg = {m: float(np.mean(v)) for m, v in by.items()}
    best = min(avg, key=avg.get)  # type: ignore[arg-type]
    rw = avg.get("rw", np.inf)
    if best != "rw" and avg[best] < 0.99 * rw:
        return best, f"'{best}' had the lowest average backtest error ({avg[best]:.2f}% vs {rw:.2f}% for the random walk)."
    return "rw", "No model beat the random walk in backtests, so the centre is held flat and the 95%-99% bands carry the signal."


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
            contrib_last = preds["driver"].contrib
    drivers_view = []
    if contrib_last:
        for name, v in sorted(contrib_last.items(), key=lambda kv: -abs(kv[1])):
            raw = float(ctx.X[name].iloc[-1])
            drivers_view.append(
                dict(
                    feature=name,
                    label="Own momentum" if name == "momentum" else GROUP_LABELS.get(name, name),
                    recent_move=raw * 100 if ctx.kinds[name] == "price" else raw,
                    unit="%" if ctx.kinds[name] == "price" else "pp",
                    pull_pct=float((np.exp(v) - 1) * 100),
                )
            )

    terminal = path[-1]
    sel_rows = [r for r in (summary or []) if r["model"] == chosen]
    avg_cov95 = float(np.mean([r["coverage95"] for r in sel_rows])) if sel_rows else 0.95
    avg_cov99 = float(np.mean([r["coverage99"] for r in sel_rows])) if sel_rows else 0.99
    if 0.90 <= avg_cov95 <= 0.985 and avg_cov99 >= 0.96:
        cal_status = f"Calibrated at 95%–99% ({avg_cov95 * 100:.1f}% on 95% CI, {avg_cov99 * 100:.1f}% on 99% CI)"
    elif avg_cov95 < 0.90 or avg_cov99 < 0.96:
        cal_status = f"Slightly tight ({avg_cov95 * 100:.1f}% on 95% CI / {avg_cov99 * 100:.1f}% on 99% CI — lean on 99% tail limits)"
    else:
        cal_status = f"Conservative / wide ({avg_cov95 * 100:.1f}% on 95% CI, {avg_cov99 * 100:.1f}% on 99% CI)"

    up_w = terminal["q975"] - terminal["mid"]
    dn_w = terminal["mid"] - terminal["q025"]
    skew_ratio = up_w / dn_w if dn_w > 1e-6 else 1.0
    if skew_ratio > 1.2:
        skew_label = "Upside depreciation skew (wider tail toward weaker kwacha)"
    elif skew_ratio < 0.83:
        skew_label = "Downside appreciation skew (wider tail toward stronger kwacha)"
    else:
        skew_label = "Roughly symmetric uncertainty band"

    unit_word = "weeks" if panel.freq == "W-FRI" else "months"
    interp_bullets = [
        f"Point vs. 95%–99% Confidence: Median projection at {h_max} {unit_word} ({terminal['date']}) is {terminal['mid']:.2f} ZMW/USD "
        f"({terminal['change_pct']:+.1f}%). Plan cashflows around the 50% core ({terminal['q25']:.2f}–{terminal['q75']:.2f}), "
        f"set standard treasury limits using the 95% CI ({terminal['q025']:.2f}–{terminal['q975']:.2f}), and stress-test against the 99% CI ({terminal['q005']:.2f}–{terminal['q995']:.2f}).",
        f"Tail Asymmetry (95% Band): {skew_label} — +{up_w:.2f} ZMW upside tail vs -{dn_w:.2f} ZMW downside tail around median.",
        f"Empirical Calibration: {cal_status}.",
    ]

    return dict(
        model=chosen,
        model_reason=why,
        last_date=last_date.strftime("%Y-%m-%d"),
        last_level=last,
        horizon=h_max,
        freq=panel.freq,
        path=path,
        drivers=drivers_view,
        drivers_used=ctx.drivers_used,
        excluded=ctx.excluded,
        lookback_periods=ctx.k,
        interpretation=dict(
            calibration_status=cal_status,
            band_skew_label=skew_label,
            bullets=interp_bullets,
        ),
    )

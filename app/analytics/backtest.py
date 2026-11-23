"""Rolling-origin backtests: the track record shown beside every forecast.

At each historical origin the models are re-fitted using only data available then (no look-ahead),
forecast h periods ahead, and compared with what actually happened. Reported per model and horizon:
accuracy (MAE, RMSE, bias), directional hit-rate, interval calibration (does the 90% band contain the
outcome ~90% of the time?), and skill vs the random walk with a HAC t-statistic so luck can be told
from a real edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .forecast import MODELS, ForecastContext
from .panel import Panel


def run_backtest(panel: Panel, horizons=(1, 4, 8, 12), min_train: int = 156, step: int = 2) -> dict:
    ctx = ForecastContext(panel)
    n = ctx.n
    rows = []
    for h in horizons:
        for t in range(min_train, n - h, step):
            try:
                preds = ctx.predict(t, h)
            except ValueError:
                continue
            actual = float(ctx.ylog[t + h] - ctx.ylog[t])
            if np.isnan(actual):
                continue
            for m in MODELS:
                p = preds[m]
                rows.append(dict(h=h, t=t, date=ctx.index[t], model=m, actual=actual, mid=p.mid,
                                 q05=p.mid + p.q[0.05], q25=p.mid + p.q[0.25],
                                 q75=p.mid + p.q[0.75], q95=p.mid + p.q[0.95]))
    if not rows:
        return dict(summary=[], track=[], n_origins=0)
    df = pd.DataFrame(rows)
    df["err"] = df["actual"] - df["mid"]
    summary = []
    for (h, m), g in df.groupby(["h", "model"]):
        rw = df[(df.h == h) & (df.model == "rw")].set_index("t")
        gg = g.set_index("t")
        mae = float(gg["err"].abs().mean() * 100)
        mae_rw = float(rw["err"].abs().mean() * 100)
        moved = gg[gg["mid"].abs() > 1e-9]
        dir_acc = float((np.sign(moved["mid"]) == np.sign(moved["actual"])).mean()) if len(moved) else None
        row = dict(h=int(h), model=m, n=int(len(gg)), mae_pct=mae, rmse_pct=float(np.sqrt((gg["err"] ** 2).mean()) * 100),
                   bias_pct=float(gg["err"].mean() * 100), direction_hit=dir_acc,
                   coverage90=float(((gg.actual >= gg.q05) & (gg.actual <= gg.q95)).mean()),
                   coverage50=float(((gg.actual >= gg.q25) & (gg.actual <= gg.q75)).mean()),
                   width90_pct=float((gg.q95 - gg.q05).mean() * 100),
                   skill_vs_rw=float(1 - mae / mae_rw) if mae_rw > 0 else None, dm_t=None)
        if m != "rw":
            d = (rw["err"].abs() - gg["err"].abs()).dropna()   # >0 means model better than RW
            if len(d) > 20 and d.std() > 0:
                fit = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": int(h)})
                row["dm_t"] = float(fit.tvalues[0])
        summary.append(row)

    # track record for the 4-period horizon, for charting
    track = []
    g4 = df[(df.h == min(horizons, key=lambda x: abs(x - 4))) & (df.model == "blend")]
    for _, r in g4.iterrows():
        track.append(dict(date=r["date"].strftime("%Y-%m-%d"), actual_pct=float((np.exp(r["actual"]) - 1) * 100),
                          mid_pct=float((np.exp(r["mid"]) - 1) * 100),
                          lo_pct=float((np.exp(r["q05"]) - 1) * 100), hi_pct=float((np.exp(r["q95"]) - 1) * 100)))
    return dict(summary=summary, track=track, n_origins=int(df["t"].nunique()),
                period=[ctx.index[min_train].strftime("%Y-%m-%d"), ctx.index[-1].strftime("%Y-%m-%d")],
                horizons=list(horizons), min_train=min_train)

"""Rolling-origin out-of-sample backtest verifying 50%, 90%, 95%, and 99% empirical coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .forecast import MODELS, ForecastContext
from .panel import Panel


def run_backtest(
    panel: Panel,
    horizons=(1, 4, 8, 12),
    min_train: int | None = None,
    step: int | None = None,
) -> dict:
    ctx = ForecastContext(panel)
    n = ctx.n
    mt = min_train if min_train is not None else (156 if panel.freq == "W-FRI" else 60)
    st = step if step is not None else (2 if panel.freq == "W-FRI" else 1)

    rows = []
    for h in horizons:
        for t in range(mt, n - h, st):
            try:
                preds = ctx.predict(t, h)
            except ValueError:
                continue
            actual = float(ctx.ylog[t + h] - ctx.ylog[t])
            if np.isnan(actual):
                continue
            for m in MODELS:
                p = preds[m]
                rows.append(
                    dict(
                        h=h,
                        t=t,
                        date=ctx.index[t],
                        model=m,
                        actual=actual,
                        mid=p.mid,
                        q005=p.mid + p.q[0.005],
                        q025=p.mid + p.q[0.025],
                        q05=p.mid + p.q[0.05],
                        q25=p.mid + p.q[0.25],
                        q75=p.mid + p.q[0.75],
                        q95=p.mid + p.q[0.95],
                        q975=p.mid + p.q[0.975],
                        q995=p.mid + p.q[0.995],
                    )
                )

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
        row = dict(
            h=int(h),
            model=m,
            n=int(len(gg)),
            mae_pct=mae,
            rmse_pct=float(np.sqrt((gg["err"] ** 2).mean()) * 100),
            bias_pct=float(gg["err"].mean() * 100),
            direction_hit=dir_acc,
            coverage50=float(((gg.actual >= gg.q25) & (gg.actual <= gg.q75)).mean()),
            coverage90=float(((gg.actual >= gg.q05) & (gg.actual <= gg.q95)).mean()),
            coverage95=float(((gg.actual >= gg.q025) & (gg.actual <= gg.q975)).mean()),
            coverage99=float(((gg.actual >= gg.q005) & (gg.actual <= gg.q995)).mean()),
            width90_pct=float((gg.q95 - gg.q05).mean() * 100),
            width95_pct=float((gg.q975 - gg.q025).mean() * 100),
            skill_vs_rw=float(1 - mae / mae_rw) if mae_rw > 0 else None,
            dm_t=None,
        )
        if m != "rw":
            d = (rw["err"].abs() - gg["err"].abs()).dropna()
            if len(d) > 20 and d.std() > 0:
                fit = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": int(h)})
                row["dm_t"] = float(fit.tvalues[0])
        summary.append(row)

    track = []
    g4 = df[(df.h == min(horizons, key=lambda x: abs(x - 4))) & (df.model == "blend")]
    for _, r in g4.iterrows():
        track.append(
            dict(
                date=r["date"].strftime("%Y-%m-%d"),
                actual_pct=float((np.exp(r["actual"]) - 1) * 100),
                mid_pct=float((np.exp(r["mid"]) - 1) * 100),
                lo_pct=float((np.exp(r["q025"]) - 1) * 100),
                hi_pct=float((np.exp(r["q975"]) - 1) * 100),
            )
        )

    return dict(
        summary=summary,
        track=track,
        n_origins=int(df["t"].nunique()),
        period=[ctx.index[mt].strftime("%Y-%m-%d"), ctx.index[-1].strftime("%Y-%m-%d")],
        horizons=list(horizons),
        min_train=mt,
    )

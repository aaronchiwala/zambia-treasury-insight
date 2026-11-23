"""Driver attribution: *what moved the kwacha, and by how much.*

Method: OLS of USD/ZMW log-returns on contemporaneous driver returns (Newey-West/HAC standard errors).
Contributions over any window = beta x cumulative driver move; the remainder is reported as
"unexplained" so the decomposition always adds up exactly. Language is deliberately associative:
this identifies what *coincided with* and was *statistically associated with* moves — not proof of cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import GROUP_LABELS, PERIODS_PER_YEAR, TARGET
from .panel import Panel

Z95 = 1.959964


@dataclass
class DriverFit:
    drivers: list[str]
    params: pd.Series               # includes 'const'
    se: pd.Series
    pvalues: pd.Series
    cov: pd.DataFrame
    r2: float
    n: int
    resid: pd.Series
    resid_sd: float
    sample_start: pd.Timestamp | None
    sample_end: pd.Timestamp | None
    excluded: list[dict] = field(default_factory=list)

    def beta(self, d: str) -> float:
        return float(self.params.get(d, 0.0))


def _select_common_sample(df: pd.DataFrame, drivers: list[str], min_obs: int):
    """Drop the most recently-starting driver until the common sample is long enough."""
    excluded: list[dict] = []
    keep = [d for d in drivers if df[d].notna().any()]
    for d in drivers:
        if d not in keep:
            excluded.append(dict(driver=d, reason="no data"))
    while keep:
        sub = df[[TARGET] + keep].dropna()
        if len(sub) >= min_obs:
            return sub, keep, excluded
        # drop the driver with the shortest history
        worst = max(keep, key=lambda d: df[d].first_valid_index() or pd.Timestamp.max)
        n_have = int(df[[TARGET, worst]].dropna().shape[0])
        excluded.append(dict(driver=worst, reason=f"insufficient overlapping history ({n_have} periods; need {min_obs})"))
        keep.remove(worst)
    return df[[TARGET]].dropna(), [], excluded


def fit_driver_model(panel: Panel, window: int | None = None, min_obs: int = 104,
                     end: pd.Timestamp | None = None, drivers: list[str] | None = None) -> DriverFit:
    df = panel.returns.copy()
    if end is not None:
        df = df.loc[:end]
    if window:
        df = df.tail(window)
    cand = [d for d in (drivers or panel.drivers) if d in df.columns]
    min_obs = min(min_obs, max(30, int(0.8 * len(df))))
    sub, keep, excluded = _select_common_sample(df, cand, min_obs)
    y = sub[TARGET]
    if len(sub) < 30:
        raise ValueError(f"Not enough overlapping data to fit a model ({len(sub)} periods).")
    X = sm.add_constant(sub[keep], has_constant="add") if keep else pd.DataFrame({"const": np.ones(len(sub))}, index=sub.index)
    maxlags = 4 if panel.freq == "W-FRI" else 2
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    resid = res.resid
    return DriverFit(
        drivers=keep, params=res.params, se=res.bse, pvalues=res.pvalues,
        cov=pd.DataFrame(res.cov_params(), index=res.params.index, columns=res.params.index),
        r2=float(res.rsquared), n=int(res.nobs), resid=resid,
        resid_sd=float(resid.std(ddof=len(res.params))), sample_start=sub.index[0], sample_end=sub.index[-1],
        excluded=excluded,
    )


def _nearest_events(date: pd.Timestamp, events: list[dict], days: int = 10) -> list[dict]:
    out = []
    for e in events:
        d = pd.Timestamp(e["date"])
        if abs((d - date).days) <= days:
            out.append(e)
    return out


def explain_period(panel: Panel, fit: DriverFit, start: str, end: str, events: list[dict] | None = None) -> dict:
    events = events or []
    lv = panel.levels[TARGET].dropna()
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
    if e_ts <= s_ts:
        raise ValueError("end must be after start")
    s_idx = lv.index.asof(s_ts)
    e_idx = lv.index.asof(e_ts)
    if pd.isna(s_idx) or pd.isna(e_idx) or e_idx <= s_idx:
        raise ValueError("Selected window falls outside the available data or contains less than one period.")
    sel = panel.returns.loc[(panel.returns.index > s_idx) & (panel.returns.index <= e_idx)]
    y = sel[TARGET]
    actual = float(y.sum())  # log points
    n = len(sel)

    contribs = {}
    for d in fit.drivers:
        x = sel[d].fillna(0.0)
        contribs[d] = fit.beta(d) * float(x.sum())
    alpha = fit.beta("const") * n
    explained = sum(contribs.values())
    unexplained = actual - explained - alpha

    lvl0, lvl1 = float(lv.loc[s_idx]), float(lv.loc[e_idx])
    pct = (lvl1 / lvl0 - 1) * 100

    rows = []
    for d in sorted(contribs, key=lambda k: -abs(contribs[k])):
        xsum = float(sel[d].sum())
        rows.append(dict(
            driver=d, label=GROUP_LABELS.get(d, d), contribution_pp=contribs[d] * 100,
            driver_move=xsum * 100 if panel.kinds[d] == "price" else xsum,
            driver_move_unit="%" if panel.kinds[d] == "price" else "pp",
            beta=fit.beta(d), p_value=float(fit.pvalues[d]), significant=bool(fit.pvalues[d] < 0.05),
        ))

    # biggest individual periods and what lined up with them
    big = y.reindex(y.abs().sort_values(ascending=False).index).head(5)
    top_periods = []
    for d, v in big.items():
        parts = {k: fit.beta(k) * float(sel.loc[d, k]) if pd.notna(sel.loc[d, k]) else 0.0 for k in fit.drivers}
        dom = max(parts, key=lambda k: abs(parts[k])) if parts else None
        resid_part = float(v) - sum(parts.values()) - fit.beta("const")
        top_periods.append(dict(
            date=d.strftime("%Y-%m-%d"), move_pct=float(v) * 100,
            dominant_driver=GROUP_LABELS.get(dom, dom) if dom and abs(parts[dom]) > abs(resid_part) else "unexplained by model",
            events=[e["title"] for e in _nearest_events(d, events)],
        ))

    in_window = [e for e in events if s_idx - pd.Timedelta(days=7) <= pd.Timestamp(e["date"]) <= e_idx]
    res = dict(
        start=s_idx.strftime("%Y-%m-%d"), end=e_idx.strftime("%Y-%m-%d"), periods=n,
        level_start=lvl0, level_end=lvl1, change_pct=pct, log_change_pp=actual * 100,
        drivers=rows, drift_pp=alpha * 100, unexplained_pp=unexplained * 100,
        explained_share=(None if abs(actual) < 1e-9 else float((explained + alpha) / actual)),
        model=dict(r2=fit.r2, n=fit.n, sample=[fit.sample_start.strftime("%Y-%m-%d"), fit.sample_end.strftime("%Y-%m-%d")],
                   excluded=fit.excluded),
        top_periods=top_periods, events=in_window,
    )
    res["narrative"] = _narrative(res, panel.freq)
    return res


def _fmt(x: float, d: int = 1) -> str:
    return f"{x:+.{d}f}"


def _narrative(r: dict, freq: str) -> list[str]:
    unit = "weeks" if freq == "W-FRI" else "months"
    direction = "weakened" if r["change_pct"] > 0 else "strengthened"
    out = [
        f"Between {r['start']} and {r['end']} ({r['periods']} {unit}) the kwacha {direction} by {abs(r['change_pct']):.1f}% "
        f"(USD/ZMW {r['level_start']:.2f} → {r['level_end']:.2f}; a higher number means a weaker kwacha)."
    ]
    sig = [d for d in r["drivers"] if d["significant"] and abs(d["contribution_pp"]) >= 0.25]
    if sig:
        bits = []
        for d in sig[:4]:
            word = "pushed USD/ZMW up (weaker kwacha)" if d["contribution_pp"] > 0 else "pushed USD/ZMW down (stronger kwacha)"
            bits.append(f"{d['label']} ({_fmt(d['driver_move'])}{d['driver_move_unit']}) {word} by about {abs(d['contribution_pp']):.1f}pp")
        out.append("Statistically significant associations: " + "; ".join(bits) + ".")
    else:
        out.append("No driver in the model shows a statistically significant, material contribution in this window.")
    if abs(r["unexplained_pp"]) >= 1.0:
        out.append(f"About {abs(r['unexplained_pp']):.1f}pp of the move is not explained by the tracked drivers "
                   "(domestic policy, sentiment, flows, debt/programme news, or data not yet in the model).")
    if r["drift_pp"] and abs(r["drift_pp"]) >= 1.0:
        out.append(f"The model's baseline drift accounts for {_fmt(r['drift_pp'])}pp over the window.")
    if r["events"]:
        out.append("Events that coincided with the window: " + "; ".join(f"{e['date']} {e['title']}" for e in r["events"][:5]) + ".")
    if r["model"]["excluded"]:
        ex = ", ".join(f"{e['driver']}" for e in r["model"]["excluded"])
        out.append(f"Not included for lack of data: {ex}. Adding history for these would sharpen the explanation.")
    out.append(f"Model fit: R² {r['model']['r2']:.2f} on {r['model']['n']} periods. "
               "These are associations, not proven causes; read alongside the event markers.")
    return out


def shock_periods(panel: Panel, fit: DriverFit, events: list[dict], z: float = 3.0) -> list[dict]:
    """Periods where the move was far larger than the drivers explain -> candidate event-driven shocks."""
    r = fit.resid
    zs = r / fit.resid_sd
    out = []
    for d, zv in zs[zs.abs() >= z].items():
        out.append(dict(date=d.strftime("%Y-%m-%d"), z=float(zv), unexplained_pct=float(r.loc[d] * 100),
                        events=[e["title"] for e in _nearest_events(d, events, 10)]))
    return out


def rolling_betas(panel: Panel, fit: DriverFit, window: int = 104) -> dict:
    from statsmodels.regression.rolling import RollingOLS
    if not fit.drivers:
        return {}
    sub = panel.returns[[TARGET] + fit.drivers].dropna()
    if len(sub) < window + 10:
        return {}
    X = sm.add_constant(sub[fit.drivers])
    rr = RollingOLS(sub[TARGET], X, window=window).fit()
    p = rr.params.dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in p.index], "betas": {c: p[c].round(4).tolist() for c in fit.drivers}}

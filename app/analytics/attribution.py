"""Driver attribution: what moved USD/ZMW in a window, and how much of the move lines up with each driver.

Methodology: HAC-robust OLS on weekly (or monthly) log-returns of prices and first differences of rates,
over the longest common history of available drivers. Includes trailing-stale recency guard, 95%-99% HAC
confidence markers, Post-2022 IMF ECF regime comparison, and plain-English interpretation bullets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import GROUP_LABELS, TARGET
from .panel import Panel

Z95 = 1.959964


@dataclass
class DriverFit:
    drivers: list[str]
    params: pd.Series  # includes 'const'
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
    """Drop trailing-stale or short-history drivers until the common sample is current and long enough."""
    excluded: list[dict] = []
    target_last = df[TARGET].last_valid_index()
    keep: list[str] = []

    for d in drivers:
        if d not in df.columns or not df[d].notna().any():
            excluded.append(dict(driver=d, reason="no data"))
            continue
        last_d = df[d].last_valid_index()
        if target_last is not None and last_d is not None:
            lag_days = int((target_last - last_d).days)
            if lag_days > 120:
                excluded.append(
                    dict(
                        driver=d,
                        reason=f"series ends {last_d.strftime('%Y-%m-%d')} ({lag_days}d behind {target_last.strftime('%Y-%m-%d')}); excluded to keep current history",
                    )
                )
                continue
        keep.append(d)

    while keep:
        sub = df[[TARGET] + keep].dropna()
        if len(sub) >= min_obs:
            return sub, keep, excluded
        worst = max(keep, key=lambda d: df[d].first_valid_index() or pd.Timestamp.max)
        n_have = int(df[[TARGET, worst]].dropna().shape[0])
        excluded.append(dict(driver=worst, reason=f"insufficient overlapping history ({n_have} periods; need {min_obs})"))
        keep.remove(worst)
    return df[[TARGET]].dropna(), [], excluded


def fit_driver_model(
    panel: Panel,
    window: int | None = None,
    min_obs: int | None = None,
    end: pd.Timestamp | None = None,
    drivers: list[str] | None = None,
) -> DriverFit:
    if min_obs is None:
        min_obs = 104 if panel.freq == "W-FRI" else 48
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
        drivers=keep,
        params=res.params,
        se=res.bse,
        pvalues=res.pvalues,
        cov=pd.DataFrame(res.cov_params(), index=res.params.index, columns=res.params.index),
        r2=float(res.rsquared),
        n=int(res.nobs),
        resid=resid,
        resid_sd=float(resid.std(ddof=len(res.params))),
        sample_start=sub.index[0],
        sample_end=sub.index[-1],
        excluded=excluded,
    )


def _nearest_events(date: pd.Timestamp, events: list[dict], days: int = 10) -> list[dict]:
    out = []
    for e in events:
        d = pd.Timestamp(e["date"])
        if abs((d - date).days) <= days:
            out.append(e)
    return out


def explain_period(
    panel: Panel,
    fit: DriverFit,
    start: str,
    end: str,
    events: list[dict] | None = None,
) -> dict:
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
        pv = float(fit.pvalues[d])
        conf = "99%" if pv < 0.01 else ("95%" if pv < 0.05 else "n.s.")
        rows.append(
            dict(
                driver=d,
                label=GROUP_LABELS.get(d, d),
                contribution_pp=contribs[d] * 100,
                driver_move=xsum * 100 if panel.kinds[d] == "price" else xsum,
                driver_move_unit="%" if panel.kinds[d] == "price" else "pp",
                beta=fit.beta(d),
                p_value=pv,
                significant=bool(pv < 0.05),
                confidence=conf,
            )
        )

    big = y.reindex(y.abs().sort_values(ascending=False).index).head(5)
    top_periods = []
    for d, v in big.items():
        parts = {k: fit.beta(k) * float(sel.loc[d, k]) if pd.notna(sel.loc[d, k]) else 0.0 for k in fit.drivers}
        dom = max(parts, key=lambda k: abs(parts[k])) if parts else None
        resid_part = float(v) - sum(parts.values()) - fit.beta("const")
        top_periods.append(
            dict(
                date=d.strftime("%Y-%m-%d"),
                move_pct=float(v) * 100,
                dominant_driver=(
                    GROUP_LABELS.get(dom, dom) if dom and abs(parts[dom]) > abs(resid_part) else "unexplained by model"
                ),
                events=[e["title"] for e in _nearest_events(d, events)],
            )
        )

    in_window = [e for e in events if s_idx - pd.Timedelta(days=7) <= pd.Timestamp(e["date"]) <= e_idx]

    # Post-2022 IMF ECF regime comparison
    regime_comp = None
    try:
        imf_rows = panel.returns.loc[panel.returns.index >= "2022-09-01"]
        if len(imf_rows) >= 60:
            reg_fit = fit_driver_model(panel, window=len(imf_rows), min_obs=45)
            top_reg = sorted(
                [
                    dict(
                        driver=d,
                        label=GROUP_LABELS.get(d, d),
                        beta=reg_fit.beta(d),
                        p_value=float(reg_fit.pvalues[d]),
                    )
                    for d in reg_fit.drivers
                ],
                key=lambda r: r["p_value"],
            )[:4]
            regime_comp = dict(
                label="Post-2022 IMF ECF Regime",
                r2=reg_fit.r2,
                n=reg_fit.n,
                sample=[reg_fit.sample_start.strftime("%Y-%m-%d"), reg_fit.sample_end.strftime("%Y-%m-%d")],
                top_drivers=top_reg,
            )
    except Exception:  # noqa: BLE001 - optional check
        regime_comp = None

    exp_share = None if abs(actual) < 1e-9 else float((explained + alpha) / actual)
    s_start = fit.sample_start.strftime("%Y-%m-%d") if fit.sample_start is not None else ""
    s_end = fit.sample_end.strftime("%Y-%m-%d") if fit.sample_end is not None else ""
    interpretation_guide = [
        (
            "Flat window."
            if exp_share is None
            else f"Explained Share: {min(100, max(0, exp_share * 100)):.0f}% of the net move aligns with tracked macro drivers & drift; "
            f"the remaining {abs(unexplained * 100):.1f}pp reflects domestic FX liquidity, IMF headlines, or unmodeled flows."
        ),
        f"Confidence Levels (95% & 99%): Drivers marked (**) reach 99% confidence (p < 0.01); (*) marks 95% confidence (p < 0.05) under Newey-West HAC errors. "
        f"Weekly R² ({fit.r2:.2f}) is naturally modest in frontier FX; switch to Monthly (ME) to include ZamStats inflation plus SA 3M and 10Y yields.",
        (
            f"Regime check: Since Sep 2022 (IMF ECF), model R² is {regime_comp['r2']:.2f} vs {fit.r2:.2f} over {s_start}–{s_end}, "
            "showing how global and South African sensitivities evolved after restructuring."
            if regime_comp
            else "Inspect rolling 2-year betas in the Data tab to check structural breaks across 2020–2022."
        ),
    ]

    res = dict(
        start=s_idx.strftime("%Y-%m-%d"),
        end=e_idx.strftime("%Y-%m-%d"),
        freq=panel.freq,
        periods=n,
        level_start=lvl0,
        level_end=lvl1,
        change_pct=pct,
        log_change_pp=actual * 100,
        drivers=rows,
        drift_pp=alpha * 100,
        unexplained_pp=unexplained * 100,
        explained_share=exp_share,
        model=dict(r2=fit.r2, n=fit.n, sample=[s_start, s_end], excluded=fit.excluded),
        regime_comparison=regime_comp,
        interpretation_guide=interpretation_guide,
        top_periods=top_periods,
        events=in_window,
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
        out.append(
            f"About {abs(r['unexplained_pp']):.1f}pp of the move is not explained by the tracked drivers "
            "(domestic policy, sentiment, flows, debt/programme news, or data not yet in the model)."
        )
    if r["drift_pp"] and abs(r["drift_pp"]) >= 1.0:
        out.append(f"The model's baseline drift accounts for {_fmt(r['drift_pp'])}pp over the window.")
    if r["events"]:
        out.append("Events that coincided with the window: " + "; ".join(f"{e['date']} {e['title']}" for e in r["events"][:5]) + ".")
    if r["model"]["excluded"]:
        ex = ", ".join(f"{e['driver']}" for e in r["model"]["excluded"])
        out.append(f"Not included in current sample: {ex}.")
    out.append(
        f"Model fit: R² {r['model']['r2']:.2f} on {r['model']['n']} periods. "
        "These are statistical associations, not proven causes; read alongside the event markers."
    )
    return out


def shock_periods(panel: Panel, fit: DriverFit, events: list[dict], z: float = 2.5) -> list[dict]:
    """Periods where the move was far larger than the drivers explain -> candidate event-driven shocks."""
    r = fit.resid
    zs = r / fit.resid_sd
    out = []
    for d, zv in zs[zs.abs() >= z].items():
        out.append(
            dict(
                date=d.strftime("%Y-%m-%d"),
                z=float(zv),
                unexplained_pct=float(r.loc[d] * 100),
                events=[e["title"] for e in _nearest_events(d, events, 10)],
            )
        )
    return out


def rolling_betas(panel: Panel, fit: DriverFit, window: int | None = None) -> dict:
    from statsmodels.regression.rolling import RollingOLS

    if not fit.drivers:
        return {}
    w = window or (104 if panel.freq == "W-FRI" else 36)
    sub = panel.returns[[TARGET] + fit.drivers].dropna()
    if len(sub) < w + 8:
        return {}
    X = sm.add_constant(sub[fit.drivers])
    rr = RollingOLS(sub[TARGET], X, window=w).fit()
    p = rr.params.dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in p.index],
        "betas": {c: p[c].round(4).tolist() for c in fit.drivers},
    }

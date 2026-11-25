"""Driver attribution & Treasury Decision Guidance for Zambia Treasury Insight.

Methodology: HAC-robust OLS on weekly (or monthly) log-returns of prices and first differences of rates,
with optional 1-period lagged external transmission and cointegrating fair-value error correction to minimize
unexplained variance across USD/ZMW, Overnight Interbank Yields, and Inflation (CPI YoY).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import GROUP_LABELS, TARGET, TARGET_META
from .panel import Panel

Z95 = 1.959964
LAG_CANDIDATES = ("COPPER", "DXY", "USDZAR", "SA_EQUITY", "EM_BOND", "VIX", "BRENT", "POLICY_RATE", "USDZMW", "CPI")


def resolve_target_col(panel: Panel, target: str) -> str:
    if target == "CPI_YOY":
        return "CPI" if "CPI" in panel.levels.columns else "CPI_YOY"
    return target


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
    adj_r2: float = 0.0
    target: str = TARGET
    target_col: str = TARGET
    target_kind: str = "price"
    lag_drivers: list[str] = field(default_factory=list)
    has_ect: bool = False
    ect_series: pd.Series | None = None

    def beta(self, d: str) -> float:
        return float(self.params.get(d, 0.0))


def _select_common_sample(df: pd.DataFrame, drivers: list[str], min_obs: int, target_col: str = TARGET):
    """Drop trailing-stale or short-history drivers until the common sample is current and long enough."""
    excluded: list[dict] = []
    if target_col not in df.columns:
        return pd.DataFrame(), [], [dict(driver=target_col, reason="target series not available in selected frequency")]
    target_last = df[target_col].last_valid_index()
    keep: list[str] = []

    for d in drivers:
        if d == target_col:
            continue
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
        sub = df[[target_col] + keep].dropna()
        if len(sub) >= min_obs:
            return sub, keep, excluded
        worst = max(keep, key=lambda d: df[d].first_valid_index() or pd.Timestamp.max)
        n_have = int(df[[target_col, worst]].dropna().shape[0])
        excluded.append(dict(driver=worst, reason=f"insufficient overlapping history ({n_have} periods; need {min_obs})"))
        keep.remove(worst)
    return df[[target_col]].dropna(), [], excluded


def _compute_ect_series(panel: Panel, target_col: str, drivers: list[str]) -> tuple[pd.Series | None, float | None]:
    """Long-run cointegrating fair-value residual z_t = log(FX_t) - X_t @ gamma (for USDZMW)."""
    if target_col != TARGET:
        return None, None
    cands = [d for d in ("COPPER", "DXY", "USDZAR", "POLICY_RATE", "ON_INTERBANK") if d in drivers and d in panel.levels.columns]
    if len(cands) < 2:
        return None, None
    lv = panel.levels[[target_col] + cands].dropna()
    if len(lv) < 80:
        return None, None
    y_lv = np.log(lv[target_col].astype(float))
    X_cols = {}
    for c in cands:
        X_cols[c] = np.log(lv[c].astype(float)) if panel.kinds[c] == "price" else lv[c].astype(float)
    X_lv = sm.add_constant(pd.DataFrame(X_cols, index=lv.index), has_constant="add")
    fit_lv = sm.OLS(y_lv, X_lv).fit()
    resid_lv = fit_lv.resid
    # Lagged residual aligned to return index
    ect_ret = resid_lv.shift(1).reindex(panel.returns.index)
    latest_fv = float(np.exp(fit_lv.fittedvalues.iloc[-1]))
    return ect_ret, latest_fv


def fit_driver_model(
    panel: Panel,
    window: int | None = None,
    min_obs: int | None = None,
    end: pd.Timestamp | None = None,
    drivers: list[str] | None = None,
    target: str = TARGET,
    with_dynamics: bool = False,
) -> DriverFit:
    target_col = resolve_target_col(panel, target)
    if target_col not in panel.returns.columns:
        raise ValueError(f"Target {target} is not available at frequency {panel.freq}. Switch to Monthly (ME) for CPI.")
    target_kind = panel.kinds.get(target_col, TARGET_META.get(target, {}).get("kind", "price"))

    if min_obs is None:
        min_obs = 104 if panel.freq == "W-FRI" else 48
    df = panel.returns.copy()
    if end is not None:
        df = df.loc[:end]
    if window:
        df = df.tail(window)

    base_pool = drivers if drivers is not None else [c for c in panel.levels.columns if c != target_col]
    cand = [d for d in base_pool if d in df.columns and d != target_col]
    min_obs = min(min_obs, max(30, int(0.8 * len(df))))
    sub, keep, excluded = _select_common_sample(df, cand, min_obs, target_col=target_col)
    if len(sub) < 30:
        raise ValueError(f"Not enough overlapping data to fit a model ({len(sub)} periods).")

    X_dict: dict[str, pd.Series] = {d: sub[d] for d in keep}
    lag_drivers: list[str] = []
    ect_series: pd.Series | None = None
    has_ect = False

    if with_dynamics and len(sub) >= 60:
        for d in keep:
            if d in LAG_CANDIDATES:
                lag_col = f"{d}__LAG1"
                X_dict[lag_col] = panel.returns[d].shift(1).reindex(sub.index)
                lag_drivers.append(d)
        ect_full, _ = _compute_ect_series(panel, target_col, keep)
        if ect_full is not None:
            ect_sub = ect_full.reindex(sub.index)
            if ect_sub.notna().sum() >= 45:
                X_dict["FAIR_VALUE_PULL"] = ect_sub
                ect_series = ect_full
                has_ect = True

    reg_df = pd.DataFrame({target_col: sub[target_col], **X_dict}).dropna()
    if len(reg_df) < 30:
        reg_df = sub[[target_col] + keep].dropna()
        lag_drivers = []
        has_ect = False
        ect_series = None

    y = reg_df[target_col]
    feat_cols = [c for c in reg_df.columns if c != target_col]
    X = sm.add_constant(reg_df[feat_cols], has_constant="add") if feat_cols else pd.DataFrame({"const": np.ones(len(reg_df))}, index=reg_df.index)
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
        adj_r2=float(res.rsquared_adj),
        n=int(res.nobs),
        resid=resid,
        resid_sd=float(resid.std(ddof=max(1, len(res.params)))),
        sample_start=reg_df.index[0],
        sample_end=reg_df.index[-1],
        excluded=excluded,
        target=target,
        target_col=target_col,
        target_kind=target_kind,
        lag_drivers=lag_drivers,
        has_ect=has_ect,
        ect_series=ect_series,
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
    tcol = fit.target_col
    meta = TARGET_META.get(fit.target, dict(label=tcol, kind=fit.target_kind))
    is_price = fit.target_kind == "price"
    scale_pp = 100.0 if is_price else 1.0

    lv = panel.levels[tcol].dropna()
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
    if e_ts <= s_ts:
        raise ValueError("end must be after start")
    s_idx = lv.index.asof(s_ts)
    e_idx = lv.index.asof(e_ts)
    if pd.isna(s_idx) or pd.isna(e_idx) or e_idx <= s_idx:
        raise ValueError("Selected window falls outside the available data or contains less than one period.")
    sel = panel.returns.loc[(panel.returns.index > s_idx) & (panel.returns.index <= e_idx)]
    y = sel[tcol]
    actual = float(y.sum())
    n = len(sel)

    rows = []
    explained = 0.0
    for d in fit.drivers:
        x0 = sel[d].fillna(0.0)
        x0_sum = float(x0.sum())
        b0 = float(fit.params.get(d, 0.0))
        contrib = b0 * x0_sum
        b_total = b0
        var_sum = float(fit.cov.loc[d, d]) if d in fit.cov.index else 0.0

        lag_col = f"{d}__LAG1"
        if d in fit.lag_drivers and lag_col in fit.params.index:
            x1 = panel.returns[d].shift(1).reindex(sel.index).fillna(0.0)
            x1_sum = float(x1.sum())
            b1 = float(fit.params[lag_col])
            contrib += b1 * x1_sum
            b_total += b1
            c11 = float(fit.cov.loc[lag_col, lag_col]) if lag_col in fit.cov.index else 0.0
            c01 = float(fit.cov.loc[d, lag_col]) if (d in fit.cov.index and lag_col in fit.cov.columns) else 0.0
            var_sum = max(0.0, var_sum + c11 + 2.0 * c01)

        se_total = float(np.sqrt(max(0.0, var_sum)))
        pv0 = float(fit.pvalues.get(d, 1.0))
        pv1 = float(fit.pvalues.get(lag_col, 1.0)) if lag_col in fit.pvalues.index else 1.0
        # Joint significance heuristic using cumulative t if lag present
        if d in fit.lag_drivers and se_total > 1e-12:
            t_cum = abs(b_total) / se_total
            # Use stricter of pvalues or cumulative normal approx
            from math import erfc, sqrt
            pv_cum = float(erfc(t_cum / sqrt(2.0)))
            pv = min(pv0, pv1, pv_cum)
        else:
            pv = pv0

        explained += contrib
        conf = "99%" if pv < 0.01 else ("95%" if pv < 0.05 else "n.s.")
        rows.append(
            dict(
                driver=d,
                label=GROUP_LABELS.get(d, d),
                contribution_pp=contrib * scale_pp,
                driver_move=x0_sum * 100.0 if panel.kinds[d] == "price" else x0_sum,
                driver_move_unit="%" if panel.kinds[d] == "price" else "pp",
                beta=b_total,
                p_value=pv,
                significant=bool(pv < 0.05),
                confidence=conf,
            )
        )

    if fit.has_ect and fit.ect_series is not None and "FAIR_VALUE_PULL" in fit.params.index:
        ect_win = fit.ect_series.reindex(sel.index).dropna()
        if len(ect_win) > 0:
            b_ect = float(fit.params["FAIR_VALUE_PULL"])
            contrib_ect = b_ect * float(ect_win.sum())
            if abs(contrib_ect * scale_pp) >= 0.05:
                pv_ect = float(fit.pvalues["FAIR_VALUE_PULL"])
                explained += contrib_ect
                rows.append(
                    dict(
                        driver="FAIR_VALUE_PULL",
                        label="Macro Fair-Value Reversion",
                        contribution_pp=contrib_ect * scale_pp,
                        driver_move=float(ect_win.iloc[0]) * scale_pp,
                        driver_move_unit="% gap" if is_price else "pp gap",
                        beta=b_ect,
                        p_value=pv_ect,
                        significant=bool(pv_ect < 0.05),
                        confidence="99%" if pv_ect < 0.01 else ("95%" if pv_ect < 0.05 else "n.s."),
                    )
                )

    rows.sort(key=lambda r: -abs(r["contribution_pp"]))
    alpha = fit.beta("const") * n
    unexplained = actual - explained - alpha
    lvl0, lvl1 = float(lv.loc[s_idx]), float(lv.loc[e_idx])
    pct = (lvl1 / lvl0 - 1) * 100.0 if is_price else (lvl1 - lvl0)

    big = y.reindex(y.abs().sort_values(ascending=False).index).head(5)
    top_periods = []
    for d_idx, v in big.items():
        parts = {}
        for k in fit.drivers:
            val0 = float(sel.loc[d_idx, k]) if pd.notna(sel.loc[d_idx, k]) else 0.0
            part = fit.beta(k) * val0
            lag_col = f"{k}__LAG1"
            if k in fit.lag_drivers and lag_col in fit.params.index:
                v_lag = panel.returns[k].shift(1).get(d_idx, np.nan)
                if pd.notna(v_lag):
                    part += float(fit.params[lag_col]) * float(v_lag)
            parts[k] = part
        if fit.has_ect and fit.ect_series is not None and "FAIR_VALUE_PULL" in fit.params.index:
            ev = fit.ect_series.get(d_idx, np.nan)
            if pd.notna(ev):
                parts["FAIR_VALUE_PULL"] = float(fit.params["FAIR_VALUE_PULL"]) * float(ev)
        dom = max(parts, key=lambda k: abs(parts[k])) if parts else None
        resid_part = float(v) - sum(parts.values()) - fit.beta("const")
        top_periods.append(
            dict(
                date=d_idx.strftime("%Y-%m-%d"),
                move_pct=float(v) * scale_pp,
                dominant_driver=(
                    GROUP_LABELS.get(dom, dom) if dom and abs(parts[dom]) > abs(resid_part) else "Domestic liquidity / event shock"
                ),
                events=[e["title"] for e in _nearest_events(d_idx, events)],
            )
        )

    in_window = [e for e in events if s_idx - pd.Timedelta(days=7) <= pd.Timestamp(e["date"]) <= e_idx]

    regime_comp = None
    try:
        imf_rows = panel.returns.loc[panel.returns.index >= "2022-09-01"]
        if len(imf_rows) >= 48:
            reg_fit = fit_driver_model(
                panel,
                window=len(imf_rows),
                min_obs=36,
                target=fit.target,
                with_dynamics=fit.has_ect or bool(fit.lag_drivers),
            )
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
    except Exception:  # noqa: BLE001
        regime_comp = None

    exp_share = None if abs(actual) < 1e-9 else float((explained + alpha) / actual)
    s_start = fit.sample_start.strftime("%Y-%m-%d") if fit.sample_start is not None else ""
    s_end = fit.sample_end.strftime("%Y-%m-%d") if fit.sample_end is not None else ""
    interpretation_guide = [
        (
            "Flat window."
            if exp_share is None
            else f"Explained Share: {min(100, max(0, exp_share * 100)):.0f}% of the net move aligns with tracked macro drivers, lagged pass-through, fair-value reversion & drift; "
            f"the remaining {abs(unexplained * scale_pp):.1f}pp reflects domestic FX clearing, auction liquidity, or unmodeled shocks."
        ),
        f"Statistical Precision: Stars denote statistical significance under Newey-West HAC errors (** p < 0.01, * p < 0.05). "
        f"Dynamic R² ({fit.r2:.2f}, adj. {fit.adj_r2:.2f}) incorporates 1-period lags so delayed mining and portfolio transmission are captured.",
        (
            f"Regime check: Since Sep 2022 (IMF ECF), model R² is {regime_comp['r2']:.2f} vs {fit.r2:.2f} over {s_start}–{s_end}, "
            "tracking how Zambia's macro elasticities evolved after sovereign debt restructuring."
            if regime_comp
            else "Inspect rolling 2-year betas in the Data tab to check structural breaks across 2020–2022."
        ),
    ]

    res = dict(
        target=fit.target,
        target_label=meta["label"],
        target_kind=fit.target_kind,
        start=s_idx.strftime("%Y-%m-%d"),
        end=e_idx.strftime("%Y-%m-%d"),
        freq=panel.freq,
        periods=n,
        level_start=lvl0,
        level_end=lvl1,
        change_pct=pct,
        log_change_pp=actual * scale_pp,
        drivers=rows,
        drift_pp=alpha * scale_pp,
        unexplained_pp=unexplained * scale_pp,
        explained_share=exp_share,
        model=dict(r2=fit.r2, adj_r2=fit.adj_r2, n=fit.n, sample=[s_start, s_end], excluded=fit.excluded),
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
    if r["target"] == TARGET:
        direction = "weakened" if r["change_pct"] > 0 else "strengthened"
        out = [
            f"Between {r['start']} and {r['end']} ({r['periods']} {unit}) the kwacha {direction} by {abs(r['change_pct']):.1f}% "
            f"(USD/ZMW {r['level_start']:.2f} → {r['level_end']:.2f}; a higher number means a weaker kwacha)."
        ]
    else:
        direction = "rose" if r["change_pct"] > 0 else "fell"
        out = [
            f"Between {r['start']} and {r['end']} ({r['periods']} {unit}) {r['target_label']} {direction} by {abs(r['change_pct']):.2f}pp "
            f"({r['level_start']:.2f}% → {r['level_end']:.2f}%)."
        ]

    sig = [d for d in r["drivers"] if d["significant"] and abs(d["contribution_pp"]) >= 0.25]
    if sig:
        bits = []
        for d in sig[:4]:
            if r["target"] == TARGET:
                word = "pushed USD/ZMW up (weaker kwacha)" if d["contribution_pp"] > 0 else "pushed USD/ZMW down (stronger kwacha)"
            else:
                word = f"lifted {r['target_label']}" if d["contribution_pp"] > 0 else f"lowered {r['target_label']}"
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
    if fit.resid_sd <= 1e-12:
        return []
    zs = r / fit.resid_sd
    scale_pp = 100.0 if fit.target_kind == "price" else 1.0
    out = []
    for d, zv in zs[zs.abs() >= z].items():
        out.append(
            dict(
                date=d.strftime("%Y-%m-%d"),
                z=float(zv),
                unexplained_pct=float(r.loc[d] * scale_pp),
                events=[e["title"] for e in _nearest_events(d, events, 10)],
            )
        )
    return out


def rolling_betas(panel: Panel, fit: DriverFit, window: int | None = None) -> dict:
    from statsmodels.regression.rolling import RollingOLS

    if not fit.drivers:
        return {}
    w = window or (104 if panel.freq == "W-FRI" else 36)
    sub = panel.returns[[fit.target_col] + fit.drivers].dropna()
    if len(sub) < w + 8:
        return {}
    X = sm.add_constant(sub[fit.drivers])
    rr = RollingOLS(sub[fit.target_col], X, window=w).fit()
    p = rr.params.dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in p.index],
        "betas": {c: p[c].round(4).tolist() for c in fit.drivers},
    }


def compute_treasury_signals(panel: Panel, load_series_fn, issues: dict | None = None) -> dict:
    """Synthesize appreciation vs depreciation signals & actionable treasury guidance."""
    lvls = panel.levels[TARGET].dropna()
    rets = panel.returns[TARGET].dropna()
    spot = float(lvls.iloc[-1])
    asof = lvls.index[-1].strftime("%Y-%m-%d")

    ma12w = float(lvls.iloc[-12:].mean()) if len(lvls) >= 12 else None
    ma26w = float(lvls.iloc[-26:].mean()) if len(lvls) >= 26 else None
    vol12w = float(rets.iloc[-12:].std(ddof=1) * np.sqrt(52) * 100.0) if len(rets) >= 12 else None
    vol52w = float(rets.iloc[-52:].std(ddof=1) * np.sqrt(52) * 100.0) if len(rets) >= 52 else None

    fair_value_mid = None
    misalignment_pct = None
    try:
        _, fv = _compute_ect_series(panel, TARGET, panel.drivers)
        if fv and spot > 0:
            fair_value_mid = fv
            misalignment_pct = (spot / fv - 1.0) * 100.0
    except Exception:  # noqa: BLE001
        pass

    def _chg(sid: str, days: int, kind: str = "price"):
        s = load_series_fn(sid)
        if s.empty or len(s) < 2:
            return None, None
        last = float(s.iloc[-1])
        ref = s.loc[: s.index[-1] - pd.Timedelta(days=days)]
        if ref.empty:
            return last, None
        r = float(ref.iloc[-1])
        if kind == "rate":
            return last, last - r
        return last, (last / r - 1.0) * 100.0 if r else None

    cu_last, cu_4w = _chg("COPPER_USD", 28, "price")
    _, cu_12w = _chg("COPPER_USD", 84, "price")
    cu_basis = "COMEX daily"
    if cu_last is None or cu_12w is None:
        # Daily feed not loaded: fall back to the monthly IMF series so the monitor still says something useful.
        m_last, m_3m = _chg("COPPER_USD_M", 90, "price")
        if m_last is not None and m_3m is not None:
            cu_last, cu_12w, cu_4w, cu_basis = m_last, m_3m, None, "IMF monthly (daily feed not loaded)"
    dxy_last, dxy_4w = _chg("DXY", 28, "price")
    vix_last, _ = _chg("VIX", 28, "price")
    _, emb_4w = _chg("EM_BOND", 28, "price")
    zar_last, zar_4w = _chg("USDZAR", 28, "price")
    _, eza_4w = _chg("SA_EQUITY", 28, "price")
    pol_rate, _ = _chg("POLICY_RATE", 28, "rate")
    on_rate, _ = _chg("ON_INTERBANK", 28, "rate")
    cpi_yoy, _ = _chg("CPI_YOY", 28, "rate")
    us10y, _ = _chg("US10Y", 28, "rate")

    real_policy_pct = (pol_rate - cpi_yoy) if (pol_rate is not None and cpi_yoy is not None) else None
    ib_spread_bps = int(round((on_rate - pol_rate) * 100)) if (on_rate is not None and pol_rate is not None) else None

    signals = []

    cu_stance = "neutral"
    if cu_12w is not None:
        if cu_12w >= 4 and (cu_4w is None or cu_4w >= 1):
            cu_stance = "appreciation"
        elif cu_12w <= -4 or (cu_4w is not None and cu_4w <= -3.5):
            cu_stance = "depreciation"
    signals.append(
        dict(
            id="copper",
            pillar="Copper & Export Inflows",
            stance=cu_stance,
            stance_label="Appreciation Support" if cu_stance == "appreciation" else ("Depreciation Risk" if cu_stance == "depreciation" else "Neutral"),
            reading=(f"${cu_last:,.0f}/t ({cu_12w:+.1f}% {'12w' if cu_4w is not None else '3m'}) · {cu_basis}" if (cu_last and cu_12w is not None)
                     else "Unavailable: no copper price loaded" + (f" ({(issues or {}).get('copper')})" if (issues or {}).get("copper") else "")),
            detail=(
                "Firm copper receipts boost mining FX supply and tax conversions into ZMW."
                if cu_stance == "appreciation"
                else ("Softer copper prices reduce expected hard-currency export inflows over coming weeks." if cu_stance == "depreciation" else "Copper is trading near recent equilibrium without a decisive export shock.")
            ),
            watch_trigger=f"Watch < ${cu_last * 0.94:,.0f}/t for FX inflow squeeze or > ${cu_last * 1.05:,.0f}/t for ZMW rally support." if cu_last else "Load daily copper (Data & quality > Retry, or upload COPPER_USD.csv) to monitor COMEX 4-week breaks.",
        )
    )

    usd_stance = "neutral"
    if dxy_4w is not None:
        if dxy_4w >= 1.5 or (vix_last is not None and vix_last >= 22) or (emb_4w is not None and emb_4w <= -2.5):
            usd_stance = "depreciation"
        elif dxy_4w <= -1.2 and (vix_last is None or vix_last < 18):
            usd_stance = "appreciation"
    signals.append(
        dict(
            id="global_usd",
            pillar="Global Dollar & EM Risk",
            stance=usd_stance,
            stance_label="Appreciation Support" if usd_stance == "appreciation" else ("Depreciation Risk" if usd_stance == "depreciation" else "Neutral"),
            reading=f"DXY {dxy_last:.1f} ({dxy_4w:+.1f}% 4w) · VIX {vix_last:.1f}" if (dxy_last and dxy_4w is not None and vix_last) else "Live feed pending",
            detail=(
                "Broad USD strength or higher global risk aversion weighs on frontier/EM currencies."
                if usd_stance == "depreciation"
                else ("Softer broad USD and contained volatility encourage offshore EM inflows." if usd_stance == "appreciation" else "Global dollar and VIX conditions are balanced.")
            ),
            watch_trigger=f"DXY sustaining above {dxy_last * 1.02:.1f} or VIX > 22 signals broad USD depreciation pressure." if dxy_last else "Watch DXY & VIX spikes.",
        )
    )

    sa_stance = "neutral"
    if zar_4w is not None:
        if zar_4w >= 2.5 or (eza_4w is not None and eza_4w <= -4):
            sa_stance = "depreciation"
        elif zar_4w <= -2.0 and (eza_4w is None or eza_4w >= 2):
            sa_stance = "appreciation"
    signals.append(
        dict(
            id="regional",
            pillar="Regional Rand & EM Sentiment",
            stance=sa_stance,
            stance_label="Appreciation Support" if sa_stance == "appreciation" else ("Depreciation Risk" if sa_stance == "depreciation" else "Neutral"),
            reading=f"USD/ZAR {zar_last:.2f} ({zar_4w:+.1f}% 4w)" if (zar_last and zar_4w is not None) else "Live feed pending",
            detail=(
                "Regional Rand weakness elevates cross-border import-cost and risk-off pressure."
                if sa_stance == "depreciation"
                else ("Stronger Rand and positive regional asset sentiment reduce cross-border FX drag." if sa_stance == "appreciation" else "Regional FX and equity markets show no acute directional pressure.")
            ),
            watch_trigger=f"USD/ZAR breaking above {zar_last * 1.035:.2f} typically coincides with regional EM risk-off." if zar_last else "Watch USD/ZAR 4-week spikes.",
        )
    )

    carry_stance = "neutral"
    if real_policy_pct is not None:
        if real_policy_pct >= 1.5 and (ib_spread_bps is None or ib_spread_bps >= -50):
            carry_stance = "appreciation"
        elif real_policy_pct < -0.5:
            carry_stance = "depreciation"
    signals.append(
        dict(
            id="monetary",
            pillar="BoZ Carry & Interbank Liquidity",
            stance=carry_stance,
            stance_label="Appreciation Support" if carry_stance == "appreciation" else ("Depreciation Risk" if carry_stance == "depreciation" else "Neutral"),
            reading=f"Real Policy {real_policy_pct:+.1f}pp · IB Spread {ib_spread_bps:+d}bp" if (real_policy_pct is not None and ib_spread_bps is not None) else "Active",
            detail=(
                f"Positive real yield ({pol_rate:.2f}% policy vs {cpi_yoy:.1f}% CPI; US10Y {us10y:.2f}%) rewards holding ZMW paper and dampens speculative USD hoarding."
                if (carry_stance == "appreciation" and pol_rate and cpi_yoy and us10y)
                else ("Negative real policy rate increases USD demand." if carry_stance == "depreciation" else "Monetary stance provides a moderate inflation cushion without acute liquidity strain.")
            ),
            watch_trigger="Track BoZ MPC rate decisions, statutory reserve ratio changes, and Overnight Interbank drifting >100bp above Policy Rate.",
        )
    )

    tech_stance = "neutral"
    if ma12w is not None and ma26w is not None:
        if spot > ma12w * 1.015 and ma12w >= ma26w:
            tech_stance = "depreciation"
        elif spot < ma12w * 0.985 and ma12w <= ma26w:
            tech_stance = "appreciation"
        elif misalignment_pct is not None and misalignment_pct > 6:
            tech_stance = "appreciation"
    signals.append(
        dict(
            id="technical",
            pillar="FX Trend & Valuation Cushion",
            stance=tech_stance,
            stance_label="Appreciation Support" if tech_stance == "appreciation" else ("Depreciation Risk" if tech_stance == "depreciation" else "Neutral"),
            reading=f"Spot {spot:.2f} vs 12w MA {ma12w:.2f}" + (f" · Macro FV ~{fair_value_mid:.2f}" if fair_value_mid else "") if ma12w else f"Spot {spot:.2f}",
            detail=(
                f"Spot trades {abs(misalignment_pct):.1f}% {'weaker' if misalignment_pct > 0 else 'stronger'} than long-run Copper/DXY/Carry fair value (~{fair_value_mid:.2f})."
                if (misalignment_pct is not None and fair_value_mid and abs(misalignment_pct) > 4)
                else "Spot is aligned with 12-week/26-week moving averages and macro fair value."
            ),
            watch_trigger=f"Close above {ma12w * 1.02:.2f} confirms upside breakout; sustained hold below {ma12w * 0.98:.2f} confirms kwacha firming." if ma12w else "Track breaks of 12w MA.",
        )
    )

    counts = {
        "appreciation": sum(1 for s in signals if s["stance"] == "appreciation"),
        "neutral": sum(1 for s in signals if s["stance"] == "neutral"),
        "depreciation": sum(1 for s in signals if s["stance"] == "depreciation"),
    }
    net = counts["depreciation"] - counts["appreciation"]
    vol_elevated = vol12w is not None and vol52w is not None and vol12w > vol52w * 1.15
    bias = "balanced"
    bias_label = "Balanced — Two-Way Flow"
    if net >= 2 or (net >= 1 and vol_elevated):
        bias = "elevated_depreciation" if (vol_elevated or net >= 3) else "depreciation"
        bias_label = "Elevated Depreciation Risk" if bias == "elevated_depreciation" else "Moderate Depreciation Pressure"
    elif net <= -1:
        bias = "appreciation"
        bias_label = "Kwacha Appreciation Tailwind"

    summary = (
        f"{counts['appreciation']} of 5 pillars favour ZMW stability or appreciation ({counts['depreciation']} depreciation risk)."
        if bias == "appreciation"
        else (
            f"Signals are mixed ({counts['appreciation']} appreciation, {counts['neutral']} neutral, {counts['depreciation']} depreciation). Expect two-way trading around {ma12w or spot:.2f}."
            if bias == "balanced"
            else f"{counts['depreciation']} of 5 pillars point toward USD/ZMW upward pressure. Prioritize payable protection over speculative waiting."
        )
    )

    playbooks = [
        dict(
            audience="Importers & USD Payables",
            posture="Layer purchases opportunistically" if bias == "appreciation" else ("Maintain core 50–60% hedge ratio" if bias == "balanced" else "Front-load & hedge 70–85% of near-term USD needs"),
            steps=[
                "Split 30–90 day USD supplier purchases into weekly tranches rather than buying a single lump sum." if bias == "appreciation" else "Keep 60-day committed USD supplier invoices pre-funded or hedged via forward contracts.",
                "Use the Forecast tab's 95% upper bound as your stress budget rate for landed import pricing.",
            ],
        ),
        dict(
            audience="Exporters & USD Earners",
            posture="Convert ZMW opex early on strength" if bias == "appreciation" else "Match conversions to ZMW cash calendar; stage discretionary sales on spikes",
            steps=[
                "Pre-convert 30–60 days of ZMW payroll and tax liabilities on kwacha strength." if bias == "appreciation" else "Convert scheduled ZMW obligations regularly and stage excess USD sales toward upper technical bands.",
            ],
        ),
        dict(
            audience="Cash, T-Bill & Bond Desk",
            posture="Real ZMW yield is positive" if (real_policy_pct is not None and real_policy_pct >= 1.0) else "Stay short duration until real yield improves",
            steps=[
                f"BoZ Policy ({pol_rate:.2f}%) exceeds CPI ({cpi_yoy:.1f}%) by {real_policy_pct:+.1f}pp — favour 182d–364d T-bills and staggered GRZ bond ladders."
                if (real_policy_pct is not None and real_policy_pct >= 1.0 and pol_rate and cpi_yoy)
                else "Favour overnight interbank and 91d T-bills while monitoring auction repricing.",
            ],
        ),
        dict(
            audience="Individuals & SMEs",
            posture="Avoid panic USD buying" if bias == "appreciation" else ("Gradual, goal-based conversion" if bias == "balanced" else "Protect near-term USD commitments"),
            steps=[
                "Match currency to obligation: keep local cash in ZMW money-market instruments and convert early only for committed 1–3 month USD bills.",
            ],
        ),
    ]

    return dict(
        asof=asof,
        spot=spot,
        bias=bias,
        bias_label=bias_label,
        summary=summary,
        counts=counts,
        metrics=dict(
            ma12w=ma12w,
            ma26w=ma26w,
            fair_value_mid=fair_value_mid,
            misalignment_pct=misalignment_pct,
            real_policy_rate_pct=real_policy_pct,
            interbank_spread_bps=ib_spread_bps,
            vol12w_ann_pct=vol12w,
            vol52w_ann_pct=vol52w,
        ),
        signals=signals,
        watchlist=[f"{s['pillar']}: {s['watch_trigger']}" for s in signals],
        playbooks=playbooks,
    )

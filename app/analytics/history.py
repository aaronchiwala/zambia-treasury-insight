"""Historical drivers: split the kwacha's history into its major swings and attribute each one.

Episodes come from a zig-zag on the weekly log level: a new episode starts when the rate reverses by more than
`threshold` from its latest extreme, so each episode is one sustained depreciation or appreciation. Each is then run
through the same attribution as the "What happened & why" tab, using ONE model fitted on the whole usable sample
(so numbers are comparable across episodes). Periods before the model's sample start are flagged, because there the
drivers have no data and the move shows up as unexplained rather than being credited to anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GROUP_LABELS, TARGET
from .attribution import DriverFit, explain_period
from .panel import Panel


def zigzag(level: pd.Series, threshold: float = 0.08) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) pairs of sustained swings in a price series."""
    lv = np.log(level.dropna())
    if len(lv) < 3:
        return []
    pivots = [lv.index[0]]
    direction = 0
    ext_i, ext_v = lv.index[0], lv.iloc[0]
    for t, v in lv.items():
        if direction >= 0 and v > ext_v:
            ext_i, ext_v = t, v
            if direction == 0 and v - lv.loc[pivots[0]] > threshold:
                direction = 1
        elif direction <= 0 and v < ext_v:
            ext_i, ext_v = t, v
            if direction == 0 and lv.loc[pivots[0]] - v > threshold:
                direction = -1
        if direction == 1 and ext_v - v > threshold:
            pivots.append(ext_i); direction = -1; ext_i, ext_v = t, v
        elif direction == -1 and v - ext_v > threshold:
            pivots.append(ext_i); direction = 1; ext_i, ext_v = t, v
    pivots.append(lv.index[-1])
    pivots = sorted(set(pivots))
    return [(a, b) for a, b in zip(pivots[:-1], pivots[1:]) if a < b]


def _slim(r: dict, covered: bool) -> dict:
    drivers = [dict(driver=d["driver"], label=d["label"], contribution_pp=d["contribution_pp"],
                    significant=d["significant"]) for d in r["drivers"]]
    top = max(drivers, key=lambda d: abs(d["contribution_pp"]), default=None)
    parts = abs(r["unexplained_pp"]) + sum(abs(d["contribution_pp"]) for d in drivers)
    return dict(start=r["start"], end=r["end"], periods=r["periods"], change_pct=r["change_pct"],
                level_start=r["level_start"], level_end=r["level_end"], log_change_pp=r["log_change_pp"],
                drivers=drivers, drift_pp=r["drift_pp"], unexplained_pp=r["unexplained_pp"],
                dominant=(top["label"] if top and abs(top["contribution_pp"]) > abs(r["unexplained_pp"]) else "Unexplained by the model"),
                unexplained_share=(abs(r["unexplained_pp"]) / parts if parts else None),
                events=[e["title"] for e in r["events"][:4]], in_model_sample=covered)


def historical_drivers(panel: Panel, fit: DriverFit, events: list[dict], threshold: float = 0.08,
                       min_periods: int = 4) -> dict:
    lv = panel.levels[TARGET].dropna()
    s0 = fit.sample_start
    episodes = []
    for a, b in zigzag(lv, threshold):
        if (lv.index.get_loc(b) - lv.index.get_loc(a)) < min_periods:
            continue
        try:
            r = explain_period(panel, fit, str(a.date()), str(b.date()), events)
        except ValueError:
            continue
        episodes.append(_slim(r, covered=bool(s0 is not None and a >= s0)))
    years = []
    for y in sorted(set(lv.index.year)):
        ys = lv[lv.index.year == y]
        prev = lv[lv.index.year == y - 1]
        start = prev.index[-1] if len(prev) else ys.index[0]
        end = ys.index[-1]
        if (end - start).days < 60:
            continue
        try:
            r = explain_period(panel, fit, str(start.date()), str(end.date()), events)
        except ValueError:
            continue
        years.append(dict(year=int(y), **_slim(r, covered=bool(s0 is not None and start >= s0))))
    covered = [e for e in episodes if e["in_model_sample"]]
    share = {}
    for e in covered:
        for d in e["drivers"]:
            share.setdefault(d["label"], []).append(abs(d["contribution_pp"]))
    avg = {k: float(np.mean(v)) for k, v in share.items()}
    avg["Unexplained by the model"] = float(np.mean([abs(e["unexplained_pp"]) for e in covered])) if covered else 0.0
    tot = sum(avg.values()) or 1.0
    summary = sorted(({"label": k, "avg_abs_pp": v, "share": v / tot} for k, v in avg.items()), key=lambda x: -x["share"])
    return dict(threshold_pct=threshold * 100, episodes=episodes, years=years, summary=summary,
                model=dict(r2=fit.r2, n=fit.n, sample=[str(fit.sample_start.date()), str(fit.sample_end.date())],
                           drivers=[GROUP_LABELS.get(d, d) for d in fit.drivers], excluded=fit.excluded),
                covered_episodes=len(covered), total_episodes=len(episodes),
                data_start=str(lv.index[0].date()), data_end=str(lv.index[-1].date()))

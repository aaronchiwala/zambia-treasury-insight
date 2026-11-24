"""Data-quality gate applied to every series before it is stored.

Principles: never silently trust a feed. Obvious unit/redenomination errors are repaired and reported;
implausible values are dropped (or the whole series rejected if it looks wrong at scale); suspicious but
possible moves are kept and flagged for a human; a refresh that would shrink stored history is refused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import JUMP_LOG, JUMP_PP, PLAUSIBLE, SERIES

REDENOM_DATE = pd.Timestamp("2013-01-01")  # kwacha rebased 1000:1 (ZMK -> ZMW) in January 2013


def validate_series(sid: str, s: pd.Series, existing_rows: int = 0,
                    today: pd.Timestamp | None = None) -> tuple[pd.Series | None, list[tuple[str, str]]]:
    """Return (clean series or None if rejected, [(level, message)]). Levels: info | warn | error."""
    issues: list[tuple[str, str]] = []
    s = pd.Series(s, dtype=float)
    s = s[np.isfinite(s)]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    today = today if today is not None else pd.Timestamp.today().normalize()

    future = s.index > today + pd.Timedelta(days=1)
    if future.any():
        issues.append(("warn", f"dropped {int(future.sum())} future-dated rows"))
        s = s[~future]

    if sid == "USDZMW":
        old = (s.index < REDENOM_DATE) & (s > 100)
        if old.any():
            s = s.copy()
            s[old] = s[old] / 1000.0
            issues.append(("info", f"rescaled {int(old.sum())} pre-2013 rows from old kwacha (ZMK) by 1/1000"))

    lo, hi = PLAUSIBLE.get(sid, (-np.inf, np.inf))
    bad = (s < lo) | (s > hi)
    if bad.any():
        if bad.mean() > 0.05:
            return None, issues + [("error", f"{int(bad.sum())} of {len(s)} values outside plausible range "
                                             f"[{lo}, {hi}] (wrong units?); series rejected")]
        issues.append(("warn", f"dropped {int(bad.sum())} out-of-range values"))
        s = s[~bad]
    if s.empty:
        return None, issues + [("error", "no usable rows")]

    if SERIES[sid]["kind"] == "price":
        step, thr = np.log(s).diff().abs(), JUMP_LOG.get(sid, 0.3)
    else:
        step, thr = s.diff().abs(), JUMP_PP.get(sid, 3.0)
    big = step[step > thr]
    if len(big):
        when = ", ".join(d.strftime("%Y-%m-%d") for d in big.index[:3])
        issues.append(("warn", f"{len(big)} unusually large one-step move(s), e.g. {when}; kept, please review"))

    if existing_rows > 50 and len(s) < 0.5 * existing_rows:
        return None, issues + [("error", f"new data has {len(s)} rows vs {existing_rows} stored; refusing to overwrite")]
    return s, issues


def summarize(issues: list[tuple[str, str]]) -> tuple[str, str]:
    """-> (status, message) for the ingest log."""
    if not issues:
        return "ok", ""
    worst = "error" if any(lv == "error" for lv, _ in issues) else "warn" if any(lv == "warn" for lv, _ in issues) else "ok"
    return worst, "; ".join(m for _, m in issues)

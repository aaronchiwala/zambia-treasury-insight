"""Event markers derived from data rather than typed by hand (so they cannot drift from the numbers)."""
from __future__ import annotations

import pandas as pd


def from_policy_rate(s: pd.Series) -> list[dict]:
    """One 'monetary' event per change in the policy-rate series (first observation is the baseline)."""
    out, prev = [], None
    for d, v in s.sort_index().items():
        if prev is not None and abs(v - prev) > 1e-9:
            bp = int(round((v - prev) * 100))
            out.append(dict(
                date=d.strftime("%Y-%m-%d"),
                title=f"BoZ {'raises' if v > prev else 'cuts'} policy rate to {v:g}% ({bp:+d}bp)",
                category="monetary", description=f"Monetary Policy Rate moved from {prev:g}% to {v:g}%.",
                source="Bank of Zambia (policy-rate series)", verified=1, is_demo=0))
        prev = v
    return out


def merge(curated: list[dict], derived: list[dict], window_days: int = 3) -> list[dict]:
    """Curated events win: skip a derived monetary event if a curated monetary one is within the window."""
    cur_dates = [pd.Timestamp(e["date"]) for e in curated if (e.get("category") or "") == "monetary"]
    keep = [e for e in derived
            if not any(abs((pd.Timestamp(e["date"]) - d).days) <= window_days for d in cur_dates)]
    return sorted(curated + keep, key=lambda e: e["date"])

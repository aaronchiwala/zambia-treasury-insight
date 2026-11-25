"""Daily copper (COMEX HG.F) from Stooq's CSV download. Second fallback after Yahoo. Best effort only."""
from __future__ import annotations

import io

import pandas as pd

from .base import BROWSER_UA, http_get, to_usd_per_tonne

URL = "https://stooq.com/q/d/l/"


def parse_csv(text: str) -> pd.Series:
    head = text.strip()[:120].lower()
    if not head.startswith("date"):
        raise ValueError(f"unexpected Stooq response: {text.strip()[:80]!r}")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df["close"], errors="coerce")
    s.index = df["date"]
    return to_usd_per_tonne(s.dropna().sort_index().astype(float))


def fetch_series() -> pd.Series:
    r = http_get(URL, params={"s": "hg.f", "i": "d"}, headers=BROWSER_UA, retries=2)
    return parse_csv(r.text)

"""Daily copper proxy (COMEX HG=F front-month) from Yahoo's unofficial chart endpoint. Best effort only."""
from __future__ import annotations

import pandas as pd

from .base import http_get, to_usd_per_tonne

URL = "https://query1.finance.yahoo.com/v8/finance/chart/HG%3DF"


def parse_chart(payload: dict) -> pd.Series:
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    close = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s").normalize()
    s = pd.Series(close, index=idx, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    return to_usd_per_tonne(s)


def fetch_series() -> pd.Series:
    r = http_get(URL, params={"range": "max", "interval": "1d"})
    return parse_chart(r.json())

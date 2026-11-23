"""Daily copper proxy (COMEX HG=F front-month) from Yahoo's unofficial chart endpoint.

Best-effort only: the endpoint is unofficial and may change or rate-limit. For production, replace with a
licensed LME feed (Bloomberg/Refinitiv/Metals API) by writing another connector that returns the
same `COPPER_USD` series. COMEX is quoted in USD/lb; converted to USD/tonne (x 2204.62).
"""
from __future__ import annotations

import pandas as pd

from .base import SeriesResult, http_get

URL = "https://query1.finance.yahoo.com/v8/finance/chart/HG%3DF"
LB_PER_TONNE = 2204.62


def parse_chart(payload: dict) -> pd.Series:
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    close = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s").normalize()
    s = pd.Series(close, index=idx, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s * LB_PER_TONNE


def fetch() -> list[SeriesResult]:
    r = http_get(URL, params={"range": "max", "interval": "1d"})
    s = parse_chart(r.json())
    return [SeriesResult("COPPER_USD", s, "Yahoo Finance HG=F (unofficial)",
                         "COMEX copper front-month futures converted to USD/tonne. Proxy for LME.")]

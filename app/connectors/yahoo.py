"""Daily COMEX copper (HG=F) and South African / EM market ETFs (EZA, EMB) from Yahoo Finance chart API."""
from __future__ import annotations

import time
from urllib.parse import quote

import pandas as pd

from .base import SeriesResult, http_get, to_usd_per_tonne

URL = "https://query1.finance.yahoo.com/v8/finance/chart/HG%3DF"
BASE_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"

# period1=2013-01-01 avoids Yahoo's range=max decimation to monthly bars
START_2013_TS = 1356998400

YAHOO_MARKET_ETFS = {
    "SA_EQUITY": (
        "EZA",
        "iShares MSCI South Africa ETF (EZA, daily USD close) — tracks South African financial & equity risk appetite.",
    ),
    "EM_BOND": (
        "EMB",
        "iShares J.P. Morgan USD Emerging Markets Bond ETF (EMB, daily close) — sovereign credit spillover benchmark.",
    ),
}


def parse_chart(payload: dict, convert_copper: bool = True) -> pd.Series:
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    close = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s").normalize()
    s = pd.Series(close, index=idx, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return to_usd_per_tonne(s) if convert_copper else s


def fetch_symbol(symbol: str = "HG=F", convert_copper: bool = True) -> pd.Series:
    endpoint = f"{BASE_CHART}{quote(symbol, safe='')}"
    now_ts = int(time.time())
    r = http_get(endpoint, params={"period1": str(START_2013_TS), "period2": str(now_ts), "interval": "1d"})
    s = parse_chart(r.json(), convert_copper=convert_copper)
    if len(s) < 250:
        r2 = http_get(endpoint, params={"range": "10y", "interval": "1d"})
        s2 = parse_chart(r2.json(), convert_copper=convert_copper)
        if len(s2) > len(s):
            return s2
    return s


def fetch_series() -> pd.Series:
    return fetch_symbol("HG=F", convert_copper=True)


def fetch_etfs() -> list[SeriesResult]:
    out = []
    for sid, (sym, note) in YAHOO_MARKET_ETFS.items():
        try:
            s = fetch_symbol(sym, convert_copper=False)
            out.append(SeriesResult(sid, s, f"Yahoo Finance {sym}", note))
        except Exception as e:  # noqa: BLE001 - isolated per series
            out.append(SeriesResult(sid, pd.Series(dtype=float), f"Yahoo Finance {sym}", f"ERROR: {e}"))
    return out

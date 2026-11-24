"""Bank of Zambia official average exchange rates, via the open Frankfurter API (provider BOZ).

Verified facts (from the provider's published OpenAPI spec and sample data, Nov 2026):
  * BoZ publishes ZMW as the base currency; asking for base=USD returns ZMW per 1 USD.
  * Monthly history in the provider starts January 2013 (the month the kwacha was rebased 1000:1).
  * A single provider serves any date range at daily granularity (the 5-year cap applies to multi-provider blends).
  * The monthly CSV export contains duplicate rows for some dates, so duplicates are removed on load.
These are *published average rates*, not intraday quotes: fine for explanation and weekly modelling, not execution.
"""
from __future__ import annotations

import pandas as pd

from .base import SeriesResult, http_get

API = "https://api.frankfurter.dev/v2/rates"


def parse_rates(records: list[dict], base: str | None = "USD") -> pd.Series:
    """Parse Frankfurter v2 records [{date, base, quote, rate}] into 'ZMW per 1 unit of base'."""
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    if base and "base" in df:
        df = df[df["base"].str.upper() == base.upper()]
    if "quote" in df:
        df = df[df["quote"].str.upper() == "ZMW"]
    df = df[pd.to_numeric(df["rate"], errors="coerce") > 0]
    if df.empty:
        return pd.Series(dtype=float)
    df = df.assign(date=pd.to_datetime(df["date"]))
    s = df.drop_duplicates("date", keep="last").set_index("date")["rate"].sort_index().astype(float)
    return s


def fetch_pair(foreign: str = "USD", start: str = "2006-01-12") -> pd.Series:
    r = http_get(API, params={"base": foreign, "quotes": "ZMW", "providers": "BOZ", "from": start})
    return parse_rates(r.json(), base=foreign)


def fetch(start: str = "2006-01-12") -> list[SeriesResult]:
    usd = fetch_pair("USD", start)
    if usd.empty:
        raise RuntimeError("Frankfurter returned no BOZ USD/ZMW data")
    return [SeriesResult("USDZMW", usd, "Bank of Zambia via frankfurter.dev",
                         "Official BoZ average rate (derived from the ZMW-base publication).")]

"""Bank of Zambia official average exchange rates, via the open Frankfurter API (provider BOZ).

BoZ publishes ZMW as the base currency; USD/ZMW is derived by inverting the published rate.
These are *published average rates*, not intraday market quotes — fine for explanation and
weekly modelling, not for execution. Replace/augment with a Bloomberg/Refinitiv feed if available.
"""
from __future__ import annotations

import pandas as pd

from .base import SeriesResult, http_get

API = "https://api.frankfurter.dev/v2/rates"


def parse_rates(records: list[dict]) -> pd.Series:
    """Parse Frankfurter v2 records [{date, base, quote, rate}] -> Series of 'ZMW per 1 unit of quote'."""
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["rate"] > 0]
    # Request is base=<foreign>, quote=ZMW so rate is already ZMW per 1 foreign unit.
    return df.set_index("date")["rate"].sort_index().astype(float)


def fetch_pair(foreign: str, start: str = "2006-01-12") -> pd.Series:
    r = http_get(API, params={"base": foreign, "quotes": "ZMW", "providers": "BOZ", "from": start})
    return parse_rates(r.json())


def fetch(start: str = "2006-01-12") -> list[SeriesResult]:
    out: list[SeriesResult] = []
    usd = fetch_pair("USD", start)
    if usd.empty:
        raise RuntimeError("Frankfurter returned no BOZ USD/ZMW data")
    out.append(SeriesResult("USDZMW", usd, "Bank of Zambia via frankfurter.dev",
                            "Official BoZ average rate (derived from ZMW-base publication)."))
    return out

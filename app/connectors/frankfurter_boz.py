"""Bank of Zambia official average exchange rates, via the open Frankfurter API (provider BOZ).

Facts taken from the provider's published OpenAPI spec and sample data (Nov 2026):
  * BoZ publishes ZMW as the base currency; asking for base=USD returns ZMW per 1 USD.
  * History starts January 2013 (the month the kwacha was rebased 1000:1).
  * Long daily ranges are computed live on the server and can hit its request deadline (HTTP 503), so the daily
    history is requested in one-year pieces. If that fails, weekly-grouped data (one small request) is used, and as
    a last resort the provider's pre-computed blended feed.
These are *published average rates*, not intraday quotes: fine for explanation and weekly modelling, not execution.
"""
from __future__ import annotations

import pandas as pd

from .base import FeedError, SeriesResult, http_get

API = "https://api.frankfurter.dev/v2/rates"
START = "2013-01-01"
MIN_DAILY_ROWS = 500
MIN_WEEKLY_ROWS = 100


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
    return df.drop_duplicates("date", keep="last").set_index("date")["rate"].sort_index().astype(float)


def _get(params: dict) -> list[dict]:
    r = http_get(API, params=params, timeout=90)
    data = r.json()
    if not isinstance(data, list):
        raise FeedError(f"{API} -> unexpected response shape: {str(data)[:150]}")
    return data


def fetch_daily(foreign: str = "USD", start: str = START, end: str | None = None) -> pd.Series:
    """Daily BoZ rates in one-year pieces; every piece must succeed and the result must have no big gaps."""
    last = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    lo, parts = pd.Timestamp(start), []
    while lo <= last:
        hi = min(lo + pd.DateOffset(years=1) - pd.Timedelta(days=1), last)
        try:
            recs = _get({"base": foreign, "quotes": "ZMW", "providers": "BOZ",
                         "from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d")})
        except FeedError as e:
            if e.status not in (404, 422):         # 'no data in this range' is tolerated; coverage check below
                raise FeedError(f"daily piece {lo.date()}..{hi.date()}: {e}", e.status) from e
            recs = []
        parts.append(parse_rates(recs, base=foreign))
        lo = hi + pd.Timedelta(days=1)
    s = pd.concat(parts) if parts else pd.Series(dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    _check_coverage(s, start, last, max_gap_days=21)
    return s


def fetch_grouped(provider: str | None = "BOZ", foreign: str = "USD", start: str = START, group: str = "week") -> pd.Series:
    """One small request: weekly (or monthly) averages dated on the first day of each period."""
    params = {"base": foreign, "quotes": "ZMW", "from": start, "group": group}
    if provider:
        params["providers"] = provider
    s = parse_rates(_get(params), base=foreign)
    _check_coverage(s, start, pd.Timestamp.today().normalize(), max_gap_days=45)
    return s


def _check_coverage(s: pd.Series, start: str, end: pd.Timestamp, max_gap_days: int) -> None:
    if s.empty:
        raise FeedError("provider returned no USD/ZMW rows")
    if (s.index.min() - pd.Timestamp(start)).days > 60:
        raise FeedError(f"history starts {s.index.min().date()}, expected about {start}")
    if (end - s.index.max()).days > 30:
        raise FeedError(f"latest rate is {s.index.max().date()}, more than 30 days old")
    gaps = s.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > max_gap_days:
        at = gaps.idxmax().date()
        raise FeedError(f"gap of {int(gaps.max())} days before {at}; refusing incomplete history")


def fetch() -> list[SeriesResult]:
    strategies = (
        ("daily", MIN_DAILY_ROWS, "Bank of Zambia daily average rates via frankfurter.dev", lambda: fetch_daily()),
        ("weekly", MIN_WEEKLY_ROWS, "Bank of Zambia WEEKLY average rates via frankfurter.dev (daily request failed)",
         lambda: fetch_grouped("BOZ")),
        ("blend", MIN_WEEKLY_ROWS, "frankfurter.dev blended WEEKLY rates (BoZ-only feed failed; check against BoZ)",
         lambda: fetch_grouped(None)),
    )
    errors: list[str] = []
    for name, min_rows, label, fn in strategies:
        try:
            s = fn()
            if len(s) < min_rows:
                raise FeedError(f"only {len(s)} rows")
        except Exception as e:  # noqa: BLE001 - any failure moves on to the next strategy
            errors.append(f"[{name}] {e}")
            continue
        note = "Official BoZ average rate (derived from the ZMW-base publication)."
        if errors:
            note += " Earlier attempts failed: " + " | ".join(errors)
        return [SeriesResult("USDZMW", s, label, note)]
    raise RuntimeError("USD/ZMW: every strategy failed. " + " | ".join(errors))

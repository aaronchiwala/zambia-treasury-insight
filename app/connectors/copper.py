"""Daily copper with a fallback chain: Yahoo HG=F -> Stooq HG.F -> (your own data/manual/COPPER_USD.csv).

Copper is the single most important external driver of the kwacha, so a silent failure here is costly:
if every source fails this raises, and the Data tab / ingest log say exactly which ones failed and why.
"""
from __future__ import annotations

from . import stooq, yahoo
from .base import SeriesResult

MIN_ROWS = 250  # about one year of daily data


def fetch() -> list[SeriesResult]:
    errors: list[str] = []
    for name, fn in (("Yahoo Finance HG=F (unofficial)", yahoo.fetch_series),
                     ("Stooq HG.F", stooq.fetch_series)):
        try:
            s = fn()
        except Exception as e:  # noqa: BLE001 - any failure moves on to the next source
            errors.append(f"{name}: {e!r}")
            continue
        if len(s) < MIN_ROWS:
            errors.append(f"{name}: only {len(s)} rows")
            continue
        return [SeriesResult("COPPER_USD", s, name,
                             "COMEX copper front-month futures in USD/tonne. Proxy for LME; replace with a licensed LME feed in production.")]
    raise RuntimeError("all daily copper sources failed: " + " | ".join(errors) +
                       ". Put a COPPER_USD.csv (date,value in USD/tonne) in data/manual/ to supply it yourself.")

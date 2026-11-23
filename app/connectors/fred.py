"""Global drivers from FRED (St. Louis Fed) public CSV endpoint — no API key needed."""
from __future__ import annotations

import io

import pandas as pd

from .base import SeriesResult, http_get

CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# our series_id -> (FRED id, note)
FRED_SERIES = {
    "DXY": ("DTWEXBGS", "Nominal Broad US Dollar Index (daily). Proxy for global dollar strength."),
    "BRENT_USD": ("DCOILBRENTEU", "Europe Brent spot price (daily)."),
    "US10Y": ("DGS10", "US 10-year constant-maturity Treasury yield (daily)."),
    "COPPER_USD_M": ("PCOPPUSDM", "Global copper price (monthly, IMF via FRED), USD per metric tonne."),
}


def parse_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col, val_col = df.columns[0], df.columns[1]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")  # FRED uses "." for missing
    df[date_col] = pd.to_datetime(df[date_col])
    return df.dropna().set_index(date_col)[val_col].astype(float).sort_index()


def fetch_one(fred_id: str) -> pd.Series:
    r = http_get(CSV, params={"id": fred_id})
    return parse_csv(r.text)


def fetch() -> list[SeriesResult]:
    out = []
    for sid, (fid, note) in FRED_SERIES.items():
        try:
            out.append(SeriesResult(sid, fetch_one(fid), f"FRED:{fid}", note))
        except Exception as e:  # keep going; ingest layer logs per-series failures
            out.append(SeriesResult(sid, pd.Series(dtype=float), f"FRED:{fid}", f"ERROR: {e}"))
    return out

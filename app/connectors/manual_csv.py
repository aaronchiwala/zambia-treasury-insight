"""Drop-in CSVs for data that has no API (BoZ policy rate, ZamStats CPI, reserves, T-bill auctions...).

Put `<SERIES_ID>.csv` in data/manual/ with columns `date,value` (ISO dates). The series must exist in
app/config.py::SERIES. Optional `# source: ...` comment line at the top is stored as provenance.
This is how the bank's own data (internal rate sheets, treasury desk exports) plugs in.
"""
from __future__ import annotations

import io

import pandas as pd

from ..config import MANUAL_DIR, SERIES
from .base import SeriesResult


def read_manual(path) -> tuple[pd.Series, str]:
    text = path.read_text(encoding="utf-8")
    sources: list[str] = []
    lines = []
    for ln in text.splitlines():
        if ln.startswith("#"):
            if ln.lower().startswith("# source:"):
                sources.append(ln.split(":", 1)[1].strip())
            continue
        lines.append(ln)
    source = " | ".join(sources) or "manual CSV"
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"].sort_index().astype(float)
    return s, source


def fetch() -> list[SeriesResult]:
    out = []
    for p in sorted(MANUAL_DIR.glob("*.csv")):
        sid = p.stem
        if sid not in SERIES:
            continue
        s, src = read_manual(p)
        out.append(SeriesResult(sid, s, f"manual: {src}", "Loaded from data/manual."))
    return out

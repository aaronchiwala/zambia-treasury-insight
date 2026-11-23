from __future__ import annotations

from dataclasses import dataclass

import httpx
import pandas as pd

UA = {"User-Agent": "zambia-treasury-insight/0.1 (research tool)"}


@dataclass
class SeriesResult:
    series_id: str
    data: pd.Series
    source: str
    notes: str = ""


def http_get(url: str, *, params: dict | None = None, timeout: float = 60.0) -> httpx.Response:
    """GET with sane defaults. Honours HTTPS_PROXY / CA bundle environment variables."""
    r = httpx.get(url, params=params, timeout=timeout, headers=UA, follow_redirects=True)
    r.raise_for_status()
    return r

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import pandas as pd

UA = {"User-Agent": "zambia-treasury-insight/0.2 (research tool)"}
LB_PER_TONNE = 2204.62


@dataclass
class SeriesResult:
    series_id: str
    data: pd.Series
    source: str
    notes: str = ""


def http_get(url: str, *, params: dict | None = None, timeout: float = 60.0, retries: int = 3) -> httpx.Response:
    """GET with retry/backoff on network errors, 429 and 5xx. Client errors (4xx) fail immediately.
    Honours HTTPS_PROXY / SSL_CERT_FILE environment variables."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, timeout=timeout, headers=UA, follow_redirects=True)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code is not None and code < 500 and code != 429:
                raise
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    assert last is not None
    raise last


def to_usd_per_tonne(s: pd.Series) -> pd.Series:
    """COMEX copper is quoted in USD/lb (~4-6); LME in USD/tonne (~8,000-13,000). Normalise to USD/tonne."""
    if s.empty:
        return s
    return s * LB_PER_TONNE if float(s.tail(500).median()) < 50 else s

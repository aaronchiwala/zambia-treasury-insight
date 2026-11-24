from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import pandas as pd

UA = {"User-Agent": "zambia-treasury-insight/0.3 (research tool)"}
LB_PER_TONNE = 2204.62


class FeedError(RuntimeError):
    """A data feed failed. The message always says which URL and what came back, so it is useful on screen."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class SeriesResult:
    series_id: str
    data: pd.Series
    source: str
    notes: str = ""


def http_get(url: str, *, params: dict | None = None, timeout: float = 60.0, retries: int = 3) -> httpx.Response:
    """GET with retry/backoff on network errors, 429 and 5xx. Client errors (4xx) fail immediately.
    Raises FeedError with the URL, HTTP status and a snippet of the response body. Honours HTTPS_PROXY."""
    last: FeedError | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, timeout=timeout, headers=UA, follow_redirects=True)
        except httpx.TransportError as e:
            last = FeedError(f"{url} -> network error: {type(e).__name__}: {e}")
        else:
            if r.status_code < 400:
                return r
            last = FeedError(f"{url} -> HTTP {r.status_code}: {r.text.strip()[:200]}", status=r.status_code)
            if r.status_code < 500 and r.status_code != 429:
                raise last
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    assert last is not None
    raise last


def to_usd_per_tonne(s: pd.Series) -> pd.Series:
    """COMEX copper is quoted in USD/lb (~4-6); LME in USD/tonne (~8,000-13,000). Normalise to USD/tonne."""
    if s.empty:
        return s
    return s * LB_PER_TONNE if float(s.tail(500).median()) < 50 else s

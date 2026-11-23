"""Synthetic demo dataset — clearly labelled, never mixed with live data.

Purpose: let the whole app (explainer, forecasts, backtests, scenarios) run before real feeds are
connected, and let tests check the engine recovers *known* parameters. The levels do NOT track real
history and must never be used for decisions. Every demo series is stored with is_demo=1 and the UI shows
a persistent banner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db
from .config import SERIES

# True data-generating parameters (the engine should recover roughly these).
TRUE_BETAS = {"COPPER": -0.35, "DXY": 0.45, "BRENT": 0.06, "POLICY_RATE": -0.006}
DEMO_EVENTS = [
    ("2016-09-12", "[DEMO] Sovereign stress shock", "sovereign", +0.11),
    ("2019-04-08", "[DEMO] Drought / power shock", "climate", +0.07),
    ("2021-08-16", "[DEMO] Election confidence rally", "political", -0.09),
    ("2022-09-05", "[DEMO] Programme approval", "programme", -0.06),
    ("2024-03-04", "[DEMO] Hydropower shortfall", "climate", +0.05),
]


def generate(seed: int = 7, start: str = "2012-01-02", end: str | None = None) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    end = end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    idx = pd.bdate_range(start, end)
    n = len(idx)

    # global drivers: correlated daily log-returns
    z = rng.standard_normal((n, 3))
    corr = np.array([[1, -0.25, 0.30], [-0.25, 1, -0.10], [0.30, -0.10, 1]])
    z = z @ np.linalg.cholesky(corr).T
    dxy_r = 0.0035 * z[:, 0]
    cu_r = 0.0125 * z[:, 1] + 0.00008
    brent_r = 0.0200 * z[:, 2]
    dxy = 100 * np.exp(np.cumsum(dxy_r))
    cu = 7500 * np.exp(np.cumsum(cu_r))
    brent = 100 * np.exp(np.cumsum(brent_r))

    # policy-rate steps at ~quarterly meetings
    rate = np.empty(n)
    cur, drate = 12.0, np.zeros(n)
    meet = set(range(30, n, 63))
    for i in range(n):
        if i in meet:
            step = rng.choice([-0.5, -0.25, 0.0, 0.0, 0.25, 0.5])
            cur = float(np.clip(cur + step, 6, 28))
            drate[i] = step
        rate[i] = cur

    # kwacha: factor model + fat-tailed, volatility-clustered noise + event jumps + mean reversion
    vol = np.empty(n)
    vol[0] = 0.0035
    for i in range(1, n):
        vol[i] = np.sqrt(0.000002 + 0.12 * (0.0035 * 0.7) ** 2 + 0.85 * vol[i - 1] ** 2) if i else 0.0035
    eps = rng.standard_t(5, n) / np.sqrt(5 / 3) * vol
    ret = (TRUE_BETAS["COPPER"] * cu_r + TRUE_BETAS["DXY"] * dxy_r + TRUE_BETAS["BRENT"] * brent_r
           + TRUE_BETAS["POLICY_RATE"] * drate + eps)
    for d, _, _, jump in DEMO_EVENTS:
        i = idx.searchsorted(pd.Timestamp(d))
        if i < n:
            ret[i] += jump
    level = np.empty(n)
    level[0] = 5.2
    for i in range(1, n):
        reversion = -0.0004 * np.log(level[i - 1] / 14.0)  # keeps demo level in a plausible band
        level[i] = level[i - 1] * np.exp(ret[i] + reversion + 0.00012)

    usdzmw = pd.Series(level, idx)

    # monthly inflation depends on lagged kwacha depreciation
    m_idx = pd.date_range(start, end, freq="ME")
    m_lvl = usdzmw.resample("ME").last().reindex(m_idx).ffill()
    dep6 = np.log(m_lvl).diff(6).fillna(0).values
    cpi = pd.Series(8 + 18 * dep6 + rng.normal(0, 0.5, len(m_idx)), m_idx).clip(3, 35)

    # monthly reserves (USD m)
    res = pd.Series(2500 * np.exp(np.cumsum(rng.normal(0.004, 0.04, len(m_idx)))), m_idx)

    return {
        "USDZMW": usdzmw,
        "COPPER_USD": pd.Series(cu, idx),
        "DXY": pd.Series(dxy, idx),
        "BRENT_USD": pd.Series(brent, idx),
        "POLICY_RATE": pd.Series(rate, idx),
        "CPI_YOY": cpi,
        "RESERVES_USD": res,
    }


def load(seed: int = 7) -> dict[str, int]:
    db.init()
    db.clear_demo()
    counts = {}
    for sid, s in generate(seed).items():
        m = SERIES[sid]
        counts[sid] = db.upsert_series(
            sid, s, name=m["name"] + " [DEMO]", unit=m["unit"], frequency=m["native"],
            source="SYNTHETIC (demo generator)", is_demo=True,
            notes="Synthetic data for development and testing. Not real market data.",
        )
    db.replace_events(
        [dict(date=d, title=t, category=c, description="Synthetic shock injected by the demo generator.",
              source="demo generator", verified=0) for d, t, c, _ in DEMO_EVENTS],
        is_demo=True,
    )
    return counts

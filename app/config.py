"""Central configuration and the series registry.

The registry is the single place that says what each series *is*: how often it is
published (`native`), how it should be transformed for modelling (`kind`), and which
modelling "group" (driver) it belongs to. Adding a new driver = add one entry here
and one connector (or a manual CSV).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.environ.get("ZTI_DB", DATA_DIR / "zti.sqlite"))
MANUAL_DIR = DATA_DIR / "manual"
EVENTS_CSV = DATA_DIR / "events.csv"

TARGET = "USDZMW"
DEFAULT_FREQ = "W-FRI"  # core modelling frequency (weekly, Friday close)

# native: D daily, E event/step series (e.g. policy rate; carried forward), W weekly,
#         M monthly, Q quarterly.
# kind:   price -> modelled as log-returns ; rate -> modelled as first differences (pp)
SERIES: dict[str, dict] = {
    "USDZMW": dict(name="USD/ZMW (kwacha per US dollar)", unit="ZMW per USD", native="D", kind="price"),
    "COPPER_USD": dict(name="Copper (daily, COMEX front-month)", unit="USD per tonne", native="D", kind="price"),
    "COPPER_USD_M": dict(name="Copper (monthly global price)", unit="USD per tonne", native="M", kind="price"),
    "DXY": dict(name="US dollar index (broad)", unit="index", native="D", kind="price"),
    "BRENT_USD": dict(name="Brent crude", unit="USD per barrel", native="D", kind="price"),
    "US10Y": dict(name="US 10-year Treasury yield", unit="percent", native="D", kind="rate"),
    "POLICY_RATE": dict(name="Bank of Zambia policy rate", unit="percent", native="E", kind="rate"),
    "CPI_YOY": dict(name="Zambia annual inflation", unit="percent", native="M", kind="rate"),
    "RESERVES_USD": dict(name="Gross international reserves", unit="USD million", native="M", kind="price"),
}

# Modelling groups: the driver name shown to users -> candidate series, finest first.
GROUPS: dict[str, list[str]] = {
    "USDZMW": ["USDZMW"],
    "COPPER": ["COPPER_USD", "COPPER_USD_M"],
    "DXY": ["DXY"],
    "BRENT": ["BRENT_USD"],
    "US10Y": ["US10Y"],
    "POLICY_RATE": ["POLICY_RATE"],
    "CPI": ["CPI_YOY"],
    "RESERVES": ["RESERVES_USD"],
}

GROUP_LABELS = {
    "COPPER": "Copper price",
    "DXY": "US dollar strength (DXY)",
    "BRENT": "Oil price (Brent)",
    "US10Y": "US 10y yield",
    "POLICY_RATE": "BoZ policy rate",
    "CPI": "Zambia inflation",
    "RESERVES": "FX reserves",
}

NATIVE_RANK = {"D": 1, "E": 1, "W": 2, "M": 3, "Q": 4}
FREQ_RANK = {"W-FRI": 2, "ME": 3}
PERIODS_PER_YEAR = {"W-FRI": 52, "ME": 12}

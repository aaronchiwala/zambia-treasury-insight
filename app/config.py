"""Series registry and modelling configuration.

Everything the app knows about series names, units, native frequencies, plausible value bounds, and which
modelling "group" (driver) it belongs to. Adding a new driver = add one entry here and one connector (or a
manual CSV).
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
    "COPPER_USD_M": dict(
        name="Copper (monthly global price)",
        unit="USD per tonne",
        native="M",
        kind="price",
        stale_days=110,  # IMF series dated the 1st, published ~5-6 weeks after month end
    ),
    "DXY": dict(name="US dollar index (broad)", unit="index", native="D", kind="price"),
    "USDZAR": dict(name="South African Rand (USD/ZAR)", unit="ZAR per USD", native="D", kind="price"),
    "SA_EQUITY": dict(name="South Africa Equity Market (EZA)", unit="USD ETF price", native="D", kind="price"),
    "EM_BOND": dict(name="EM Sovereign Bond Market (EMB)", unit="USD ETF price", native="D", kind="price"),
    "BRENT_USD": dict(name="Brent crude", unit="USD per barrel", native="D", kind="price"),
    "US10Y": dict(name="US 10-year Treasury yield", unit="percent", native="D", kind="rate"),
    "SA_10Y": dict(name="South Africa 10Y bond yield", unit="percent", native="M", kind="rate", stale_days=110),
    "SA_3M_RATE": dict(name="South Africa 3M interbank rate", unit="percent", native="M", kind="rate", stale_days=110),
    "POLICY_RATE": dict(name="Bank of Zambia policy rate", unit="percent", native="E", kind="rate"),
    "ON_INTERBANK": dict(name="Overnight interbank rate", unit="percent", native="D", kind="rate"),
    "CPI_YOY": dict(name="Zambia annual inflation", unit="percent", native="M", kind="rate"),
    "RESERVES_USD": dict(name="Gross international reserves", unit="USD million", native="M", kind="price"),
    "VIX": dict(name="VIX equity volatility index (global risk appetite)", unit="index", native="D", kind="price"),
}

# Plausible value ranges (unit-error / bad-feed guard) and "unusually large one-step move" flags.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "USDZMW": (1.0, 60.0),
    "COPPER_USD": (1500.0, 22000.0),
    "COPPER_USD_M": (1500.0, 22000.0),
    "DXY": (60.0, 200.0),
    "USDZAR": (5.0, 35.0),
    "SA_EQUITY": (10.0, 150.0),
    "EM_BOND": (40.0, 160.0),
    "BRENT_USD": (5.0, 220.0),
    "US10Y": (-1.0, 20.0),
    "SA_10Y": (2.0, 25.0),
    "SA_3M_RATE": (1.0, 25.0),
    "VIX": (5.0, 100.0),
    "POLICY_RATE": (0.0, 80.0),
    "ON_INTERBANK": (0.0, 80.0),
    "CPI_YOY": (-5.0, 100.0),
    "RESERVES_USD": (100.0, 50000.0),
}

JUMP_LOG = {
    "USDZMW": 0.15,
    "COPPER_USD": 0.15,
    "DXY": 0.05,
    "USDZAR": 0.15,
    "SA_EQUITY": 0.25,
    "EM_BOND": 0.15,
    "BRENT_USD": 0.4,
    "VIX": 0.9,
    "RESERVES_USD": 0.5,
}
JUMP_PP = {
    "US10Y": 1.0,
    "SA_10Y": 2.0,
    "SA_3M_RATE": 2.0,
    "CPI_YOY": 4.0,
    "POLICY_RATE": 10.0,
    "ON_INTERBANK": 12.0,
}

# Modelling groups: the driver name shown to users -> candidate series, finest first.
GROUPS: dict[str, list[str]] = {
    "USDZMW": ["USDZMW"],
    "COPPER": ["COPPER_USD", "COPPER_USD_M"],
    "DXY": ["DXY"],
    "USDZAR": ["USDZAR"],
    "SA_EQUITY": ["SA_EQUITY"],
    "EM_BOND": ["EM_BOND"],
    "BRENT": ["BRENT_USD"],
    "US10Y": ["US10Y"],
    "POLICY_RATE": ["POLICY_RATE"],
    "ON_INTERBANK": ["ON_INTERBANK"],
    "SA_10Y": ["SA_10Y"],
    "SA_3M_RATE": ["SA_3M_RATE"],
    "CPI": ["CPI_YOY"],
    "RESERVES": ["RESERVES_USD"],
    "VIX": ["VIX"],
}

GROUP_LABELS = {
    "COPPER": "Copper price",
    "DXY": "US dollar strength (DXY)",
    "USDZAR": "South African Rand (USD/ZAR)",
    "SA_EQUITY": "SA financial market (EZA)",
    "EM_BOND": "EM sovereign debt (EMB)",
    "BRENT": "Oil price (Brent)",
    "US10Y": "US 10y yield",
    "POLICY_RATE": "BoZ policy rate",
    "ON_INTERBANK": "Overnight interbank rate",
    "SA_10Y": "SA 10y bond yield",
    "SA_3M_RATE": "SA 3m money market rate",
    "CPI": "Zambia inflation",
    "RESERVES": "FX reserves",
    "VIX": "Global risk appetite (VIX)",
}

NATIVE_RANK = {"D": 1, "E": 1, "W": 2, "M": 3, "Q": 4}
FREQ_RANK = {"W-FRI": 2, "ME": 3}
PERIODS_PER_YEAR = {"W-FRI": 52, "ME": 12}

"""Macro and market shock analysis built on historical driver sensitivities."""
from __future__ import annotations

import numpy as np

from ..config import GROUP_LABELS, TARGET
from .attribution import DriverFit, fit_driver_model
from .panel import Panel

PRESETS = {
    "copper_crash": dict(label="Copper falls 20%", shocks={"COPPER": -20.0}),
    "copper_rally": dict(label="Copper rallies 15%", shocks={"COPPER": 15.0}),
    "usd_rally": dict(label="Global dollar rallies 5%", shocks={"DXY": 5.0}),
    "sa_spillover": dict(label="Regional FX & EM risk-off (USD/ZAR +8%, EZA -12%, EMB -5%)", shocks={"USDZAR": 8.0, "SA_EQUITY": -12.0, "EM_BOND": -5.0}),
    "oil_spike": dict(label="Oil price spikes 30%", shocks={"BRENT": 30.0}),
    "rate_cut_100": dict(label="BoZ cuts policy rate 100bp", shocks={"POLICY_RATE": -100.0}),
    "risk_off": dict(label="Risk-off: copper -15%, dollar +4%", shocks={"COPPER": -15.0, "DXY": 4.0}),
}
Z = {"80": 1.2816, "95": 1.9600, "99": 2.5758}


def _convert(panel: Panel, shocks: dict[str, float]) -> dict[str, float]:
    out = {}
    for d, v in shocks.items():
        kind = panel.kinds.get(d)
        if kind is None:
            raise ValueError(f"Driver {d!r} is not available in the current data.")
        out[d] = float(np.log1p(v / 100.0)) if kind == "price" else float(v) / 100.0
    return out


def _impact(fit: DriverFit, s: dict[str, float], horizon: int) -> tuple[float, float, dict[str, float]]:
    missing = [d for d in s if d not in fit.drivers]
    if missing:
        why = {e["driver"]: e["reason"] for e in fit.excluded}
        raise ValueError("Cannot shock " + ", ".join(f"{GROUP_LABELS.get(d, d)} ({why.get(d, 'not in model')})" for d in missing))
    vec = np.array([s.get(d, 0.0) for d in fit.drivers])
    betas = np.array([fit.beta(d) for d in fit.drivers])
    cov = fit.cov.loc[fit.drivers, fit.drivers].values
    centre = float(vec @ betas)
    sd = float(np.sqrt(max(vec @ cov @ vec, 0.0) + fit.resid_sd ** 2 * horizon))
    parts = {d: float(fit.beta(d) * s.get(d, 0.0)) for d in fit.drivers if d in s}
    return centre, sd, parts


def run_scenario(panel: Panel, shocks: dict[str, float], horizon: int = 12, window: int = 156) -> dict:
    s = _convert(panel, shocks)
    recent = fit_driver_model(panel, window=window)
    full = fit_driver_model(panel)
    c_r, sd_r, parts = _impact(recent, s, horizon)
    c_f, sd_f, _ = _impact(full, s, horizon)
    last = float(panel.levels[TARGET].iloc[-1])
    lvl = lambda x: float(last * np.exp(x))
    pct = lambda x: float((np.exp(x) - 1) * 100)
    bands = {
        k: dict(
            low_pct=pct(c_r - z * sd_r),
            high_pct=pct(c_r + z * sd_r),
            low_level=lvl(c_r - z * sd_r),
            high_level=lvl(c_r + z * sd_r),
        )
        for k, z in Z.items()
    }
    unstable = (np.sign(c_r) != np.sign(c_f) and max(abs(c_r), abs(c_f)) > 0.005) or (
        abs(c_r - c_f) > 0.5 * max(abs(c_r), abs(c_f), 1e-9) and abs(c_r - c_f) > 0.01
    )
    warn = []
    if unstable:
        warn.append(
            "Regime-sensitive: the recent-window and full-history sensitivities disagree materially. "
            "Treat the central estimate as indicative and lean on the wider risk bands."
        )
    if recent.r2 < 0.05:
        warn.append(f"Low explanatory power (R² {recent.r2:.2f}) in the recent window: sensitivities are weakly identified.")
    return dict(
        shocks=shocks,
        horizon=horizon,
        window=window,
        last_level=last,
        impact_pct=pct(c_r),
        impact_level=lvl(c_r),
        bands=bands,
        by_driver=[dict(driver=d, label=GROUP_LABELS.get(d, d), impact_pp=v * 100) for d, v in parts.items()],
        recent=dict(impact_pct=pct(c_r), r2=recent.r2, n=recent.n),
        full_history=dict(impact_pct=pct(c_f), r2=full.r2, n=full.n),
        regime_sensitive=bool(unstable),
        warnings=warn,
        reading=(
            "USD/ZMW up = weaker kwacha. Impact is relative to the no-shock baseline over the horizon; "
            "it is an estimate from historical co-movement, not a forecast of the shock itself."
        ),
        available_drivers=[
            dict(
                driver=d,
                label=GROUP_LABELS.get(d, d),
                kind=panel.kinds[d],
                unit="% change" if panel.kinds[d] == "price" else "basis points",
            )
            for d in recent.drivers
        ],
    )


def available_presets(panel: Panel, window: int = 156) -> list[dict]:
    drivers = set(fit_driver_model(panel, window=window).drivers)
    return [dict(key=k, label=v["label"], shocks=v["shocks"]) for k, v in PRESETS.items() if set(v["shocks"]) <= drivers]

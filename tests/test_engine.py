import numpy as np
import pandas as pd
import pytest

from app import db
from app.analytics import attribution, backtest, forecast, scenario
from app.analytics.panel import build_panel
from app.config import TARGET
from app.demo_data import TRUE_BETAS


@pytest.fixture(scope="module")
def panel():
    return build_panel("W-FRI")


def test_panel_skips_coarser_series_instead_of_forward_filling(panel):
    skipped = {s["series"] for s in panel.skipped}
    assert {"CPI_YOY", "RESERVES_USD"} <= skipped
    assert "CPI" not in panel.levels.columns
    assert panel.is_demo


def test_panel_drops_incomplete_trailing_period(panel):
    last_obs = db.load_series("USDZMW").index.max()
    assert panel.levels.index[-1] <= last_obs


def test_monthly_frequency_includes_monthly_drivers():
    pm = build_panel("ME")
    assert "CPI" in pm.levels.columns


def test_fit_recovers_planted_sensitivities(panel):
    fit = attribution.fit_driver_model(panel)
    for d, true in TRUE_BETAS.items():
        assert fit.beta(d) == pytest.approx(true, abs=0.07 if d != "POLICY_RATE" else 0.004), d
    assert fit.pvalues["COPPER"] < 0.01 and fit.pvalues["DXY"] < 0.01


def test_attribution_adds_up_exactly(panel):
    fit = attribution.fit_driver_model(panel)
    r = attribution.explain_period(panel, fit, "2019-01-01", "2020-06-30", db.load_events(True))
    total = sum(d["contribution_pp"] for d in r["drivers"]) + r["drift_pp"] + r["unexplained_pp"]
    assert total == pytest.approx(r["log_change_pp"], abs=1e-9)
    assert r["narrative"] and "associations" in r["narrative"][-1]
    assert r["interpretation_guide"]


def test_attribution_rejects_bad_windows(panel):
    fit = attribution.fit_driver_model(panel)
    with pytest.raises(ValueError):
        attribution.explain_period(panel, fit, "2020-06-01", "2020-01-01")


def test_injected_shock_is_flagged_and_matched_to_event(panel):
    fit = attribution.fit_driver_model(panel)
    shocks = attribution.shock_periods(panel, fit, db.load_events(True), z=3.5)
    hits = [s for s in shocks if s["date"].startswith("2016-09") and s["events"]]
    assert hits, "the +11% demo sovereign shock should surface as an unexplained, event-matched move"


def test_policy_rate_dropped_when_history_too_short():
    p = build_panel("W-FRI")
    p2 = build_panel("W-FRI")
    p2.returns.loc[: p2.returns.index[-60], "POLICY_RATE"] = np.nan
    fit = attribution.fit_driver_model(p2)
    assert "POLICY_RATE" not in fit.drivers
    assert any(e["driver"] == "POLICY_RATE" for e in fit.excluded)
    assert "POLICY_RATE" in attribution.fit_driver_model(p).drivers


def test_forecast_bands_are_ordered_and_widen(panel):
    f = forecast.make_forecast(panel, 12, "blend")
    for p in f["path"]:
        assert p["q005"] <= p["q025"] <= p["q05"] < p["q25"] <= p["mid"] <= p["q75"] < p["q95"] <= p["q975"] <= p["q995"]
    assert f["path"][-1]["q995"] - f["path"][-1]["q005"] > f["path"][0]["q995"] - f["path"][0]["q005"]


def test_backtest_intervals_are_calibrated_and_no_lookahead(panel):
    bt = backtest.run_backtest(panel, horizons=(4,), step=4)
    rw = next(r for r in bt["summary"] if r["model"] == "rw")
    assert 0.80 <= rw["coverage90"] <= 0.99
    assert 0.90 <= rw["coverage95"] <= 1.00
    assert 0.94 <= rw["coverage99"] <= 1.00
    assert 0.35 <= rw["coverage50"] <= 0.65
    ctx = forecast.ForecastContext(panel)
    s = ctx._train_slice(300, 4, 260)
    assert (s + 4 <= 300).all()


def test_auto_model_falls_back_to_rw_when_no_edge():
    summary = [dict(model="rw", mae_pct=2.0), dict(model="driver", mae_pct=2.1), dict(model="blend", mae_pct=2.0)]
    assert forecast.select_model(summary)[0] == "rw"
    summary[1]["mae_pct"] = 1.5
    assert forecast.select_model(summary)[0] == "driver"


def test_scenario_signs_make_economic_sense(panel):
    crash = scenario.run_scenario(panel, {"COPPER": -20.0}, 12, 156)
    assert crash["impact_pct"] > 0
    assert crash["bands"]["99"]["low_pct"] < crash["bands"]["95"]["low_pct"] < crash["bands"]["80"]["low_pct"] < crash["impact_pct"]
    usd = scenario.run_scenario(panel, {"DXY": 5.0}, 12, 156)
    assert usd["impact_pct"] > 0
    big = scenario.run_scenario(panel, {"COPPER": -20.0}, 12, 156)["impact_pct"]
    small = scenario.run_scenario(panel, {"COPPER": -5.0}, 12, 156)["impact_pct"]
    assert big > small > 0


def test_scenario_rejects_unavailable_driver(panel):
    with pytest.raises(ValueError):
        scenario.run_scenario(panel, {"RESERVES": -10.0}, 12, 156)


def test_history_episodes_and_years_add_up():
    from fastapi.testclient import TestClient
    from app.main import app

    j = TestClient(app).get("/api/history?threshold=8").json()
    assert j["total_episodes"] >= 5 and j["years"]
    for e in j["episodes"]:
        total = sum(d["contribution_pp"] for d in e["drivers"]) + e["drift_pp"] + e["unexplained_pp"]
        assert abs(total - e["log_change_pp"]) < 1e-6
    ends = [e["end"] for e in j["episodes"]]
    starts = [e["start"] for e in j["episodes"]]
    assert all(a >= b for a, b in zip(starts[1:], ends[:-1]))

"""Connector parsers (against recorded-style payloads) and the HTTP API."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.connectors import fred, frankfurter_boz, manual_csv, yahoo
from app.config import MANUAL_DIR
from app.main import app

client = TestClient(app)


def test_frankfurter_parser_handles_v2_records():
    recs = [dict(date="2026-10-08", base="USD", quote="ZMW", rate=19.89),
            dict(date="2026-10-09", base="USD", quote="ZMW", rate=19.95),
            dict(date="2026-10-10", base="USD", quote="ZMW", rate=0)]  # invalid rate dropped
    s = frankfurter_boz.parse_rates(recs)
    assert list(s.round(2)) == [19.89, 19.95]
    assert frankfurter_boz.parse_rates([]).empty


def test_fred_parser_drops_missing_markers():
    csv = "observation_date,DTWEXBGS\n2026-01-01,120.1\n2026-01-02,.\n2026-01-05,120.4\n"
    s = fred.parse_csv(csv)
    assert len(s) == 2 and s.iloc[-1] == pytest.approx(120.4)


def test_yahoo_parser_converts_lb_to_tonne():
    payload = {"chart": {"result": [{"timestamp": [1767225600, 1767312000],
                                     "indicators": {"quote": [{"close": [5.0, None]}]}}]}}
    s = yahoo.parse_chart(payload)
    assert len(s) == 1 and s.iloc[0] == pytest.approx(5.0 * 2204.62)


def test_manual_seed_files_parse():
    s, src = manual_csv.read_manual(MANUAL_DIR / "POLICY_RATE.csv")
    assert s.iloc[-1] == 10.75 and "Bank of Zambia" in src
    cpi, _ = manual_csv.read_manual(MANUAL_DIR / "CPI_YOY.csv")
    assert cpi.index.is_monotonic_increasing and len(cpi) == 11


def test_status_flags_demo_mode():
    r = client.get("/api/status").json()
    assert r["mode"] == "demo" and r["notes"]
    assert all(s["is_demo"] for s in r["series"])


def test_overview_and_series():
    ov = client.get("/api/overview").json()
    assert ov["cards"][0]["series_id"] == "USDZMW"
    s = client.get("/api/series", params={"ids": "USDZMW,DXY", "start": "2025-01-01"}).json()
    assert set(s) == {"USDZMW", "DXY"} and s["USDZMW"]["dates"][0] >= "2025-01-01"


def test_explain_endpoint_and_bad_input():
    r = client.get("/api/explain", params={"start": "2021-06-01", "end": "2021-12-31"})
    assert r.status_code == 200 and r.json()["narrative"]
    bad = client.get("/api/explain", params={"start": "2021-12-31", "end": "2021-06-01"})
    assert bad.status_code == 422
    assert client.get("/api/explain", params={"start": "2021-06-01", "end": "2021-12-31", "freq": "X"}).status_code == 409


def test_forecast_and_backtest_endpoints():
    f = client.get("/api/forecast", params={"h": 8}).json()
    assert len(f["path"]) == 8 and f["backtest_summary"] and f["is_demo"] is True
    assert client.get("/api/forecast", params={"model": "nope"}).status_code == 422


def test_scenario_endpoint():
    r = client.post("/api/scenario", json={"shocks": {"COPPER": -20, "DXY": 4}, "horizon": 12})
    assert r.status_code == 200 and r.json()["impact_pct"] > 0
    assert client.post("/api/scenario", json={"shocks": {"NOPE": 1}}).status_code == 422
    assert client.get("/api/scenario/presets").json()["presets"]


def test_model_and_static_page():
    m = client.get("/api/model").json()
    assert m["rolling"]["dates"] and {d["driver"] for d in m["drivers"]} >= {"COPPER", "DXY"}
    page = client.get("/")
    assert page.status_code == 200 and "Zambia Treasury Insight" in page.text

"""Seed data integrity, validation gate, derived events, retries, fallbacks, ingest, scheduler, auth."""
import base64

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db, events_derived, ingest
from app.analytics.forecast import ForecastContext
from app.analytics.panel import build_panel
from app.config import MANUAL_DIR
from app.connectors import base, copper, manual_csv, stooq, yahoo
from app.connectors.base import SeriesResult
from app.demo_data import generate
from app.main import app
from app.scheduler import Refresher
from app.validate import validate_series

client = TestClient(app)


def _series(vals, start="2020-01-03", freq="W-FRI"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq), dtype=float)


# ---------------- seed data ----------------
def test_policy_rate_seed_matches_boz_table():
    s, src = manual_csv.read_manual(MANUAL_DIR / "POLICY_RATE.csv")
    assert s.index.is_monotonic_increasing and not s.index.has_duplicates
    assert (s.index[0].date().isoformat(), s.iloc[0]) == ("2012-04-02", 9.0)
    assert (s.index[-1].date().isoformat(), s.iloc[-1]) == ("2026-09-30", 10.75)
    assert len(s) == 31 and "Monetary Policy Report" in src
    # BoZ's own 'reduced/raised ... from X' statements chain to the preceding level
    at = lambda d: float(s.loc[:d].iloc[-1])
    assert at("2017-02-21") == 15.5 and at("2017-05-16") == 14.0 and at("2019-05-21") == 9.75
    assert at("2020-05-19") == 11.5 and at("2021-11-23") == 8.5 and at("2024-11-12") == 13.5
    assert at("2025-11-11") == 14.5            # the Feb-2025 hike to 14.5 (not 19 Feb)
    assert s.diff().abs().max() <= 3.0         # biggest single move is the 300bp hike of Nov 2015


def test_cpi_seed_is_monthly_in_range_and_matches_known_prints():
    s, src = manual_csv.read_manual(MANUAL_DIR / "CPI_YOY.csv")
    assert s.index.is_monotonic_increasing and not s.index.has_duplicates
    assert all(d == d + pd.offsets.MonthEnd(0) for d in s.index)          # month-end dates
    assert s.between(5, 30).all()
    assert s.loc["2025-03-31"] == 16.5 and s.loc["2025-11-30"] == 10.9 and s.loc["2026-10-31"] == 6.3
    assert s.index[0] == pd.Timestamp("2020-02-29") and "ZamStats" in src


def test_manual_csv_collects_every_source_line():
    _, src = manual_csv.read_manual(MANUAL_DIR / "CPI_YOY.csv")
    assert " | " in src


# ---------------- validation gate ----------------
def test_redenomination_guard_rescales_old_kwacha():
    idx = pd.to_datetime(["2012-06-01", "2012-12-31", "2013-01-02", "2013-01-03"])
    s, issues = validate_series("USDZMW", pd.Series([4800.0, 5000.0, 5.1, 5.2], index=idx))
    assert s.iloc[0] == pytest.approx(4.8) and s.iloc[2] == 5.1
    assert any("ZMK" in m for _, m in issues)


def test_out_of_range_series_is_rejected_and_stray_points_dropped():
    s, issues = validate_series("USDZMW", _series([500.0] * 40))
    assert s is None and issues[-1][0] == "error"
    mostly_ok = _series([20.0] * 60 + [999.0])
    s, issues = validate_series("USDZMW", mostly_ok)
    assert len(s) == 60 and any("dropped 1" in m for _, m in issues)


def test_jump_is_flagged_not_removed():
    vals = [20.0] * 30 + [26.0] + [26.0] * 5
    s, issues = validate_series("USDZMW", _series(vals))
    assert len(s) == len(vals) and any("large one-step" in m for _, m in issues)


def test_refresh_never_shrinks_stored_history():
    s, issues = validate_series("USDZMW", _series([20.0] * 40), existing_rows=3000)
    assert s is None and "refusing to overwrite" in issues[-1][1]


def test_future_dates_dropped():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2099-01-01"])
    s, _ = validate_series("USDZMW", pd.Series([20.0, 20.1, 20.2], index=idx))
    assert len(s) == 2


# ---------------- derived events ----------------
def test_events_derived_from_policy_rate_and_merge_prefers_curated():
    s, _ = manual_csv.read_manual(MANUAL_DIR / "POLICY_RATE.csv")
    ev = events_derived.from_policy_rate(s)
    assert len(ev) == len(s) - 1
    assert ev[0]["title"] == "BoZ raises policy rate to 9.25% (+25bp)"
    assert next(e for e in ev if e["date"] == "2026-09-30")["title"].endswith("10.75% (-250bp)")
    curated = [dict(date="2026-09-30", title="curated", category="monetary")]
    merged = events_derived.merge(curated, ev)
    assert sum(1 for e in merged if e["date"] == "2026-09-30") == 1
    assert [e["date"] for e in merged] == sorted(e["date"] for e in merged)


# ---------------- connectors ----------------
def test_http_get_retries_then_succeeds_and_does_not_retry_404(monkeypatch):
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(url, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, request=httpx.Request("GET", url))
        return httpx.Response(200, text="ok", request=httpx.Request("GET", url))

    monkeypatch.setattr(base.httpx, "get", flaky)
    assert base.http_get("https://example.test/x").text == "ok" and calls["n"] == 3

    calls["n"] = 0
    monkeypatch.setattr(base.httpx, "get", lambda url, **kw: (calls.__setitem__("n", calls["n"] + 1),
                        httpx.Response(404, request=httpx.Request("GET", url)))[1])
    with pytest.raises(base.FeedError) as exc:
        base.http_get("https://example.test/missing")
    assert exc.value.status == 404 and "example.test/missing" in str(exc.value)
    assert calls["n"] == 1


def test_unit_normalisation_and_stooq_parser():
    assert base.to_usd_per_tonne(_series([4.0, 4.1])).iloc[0] == pytest.approx(4.0 * 2204.62)
    assert base.to_usd_per_tonne(_series([9000.0, 9100.0])).iloc[0] == 9000.0
    s = stooq.parse_csv("Date,Open,High,Low,Close,Volume\n2026-11-20,5.0,5.1,4.9,5.05,100\n2026-11-21,5.1,5.2,5.0,5.15,90\n")
    assert len(s) == 2 and s.iloc[-1] == pytest.approx(5.15 * 2204.62)
    with pytest.raises(ValueError):
        stooq.parse_csv("Exceeded the daily hits limit")


def test_copper_chain_falls_back_and_reports_all_failures(monkeypatch):
    good = _series(np.linspace(8000, 9000, 300), freq="B")
    monkeypatch.setattr(yahoo, "fetch_series", lambda: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(stooq, "fetch_series", lambda: good)
    res = copper.fetch()
    assert res[0].series_id == "COPPER_USD" and "Stooq" in res[0].source
    monkeypatch.setattr(stooq, "fetch_series", lambda: (_ for _ in ()).throw(RuntimeError("limit")))
    with pytest.raises(RuntimeError) as e:
        copper.fetch()
    assert "Yahoo" in str(e.value) and "Stooq" in str(e.value) and "COPPER_USD.csv" in str(e.value)


# ---------------- ingest ----------------
def _fake_sources(monkeypatch, fx_rows=400):
    fx = _series(np.linspace(5.0, 20.0, fx_rows), start="2013-01-04", freq="B")
    monkeypatch.setattr(ingest.frankfurter_boz, "fetch", lambda: [SeriesResult("USDZMW", fx, "fake BoZ")])
    monkeypatch.setattr(ingest.fred, "fetch", lambda: [SeriesResult("DXY", _series(np.linspace(90, 110, fx_rows), start="2013-01-04", freq="B"), "fake FRED")])
    monkeypatch.setattr(ingest.copper, "fetch", lambda: (_ for _ in ()).throw(RuntimeError("both blocked")))
    return fx


def test_ingest_dry_run_writes_nothing(tmp_db, monkeypatch):
    _fake_sources(monkeypatch)
    summary = ingest.run_live(dry=True, quiet=True)
    assert "USDZMW" in summary["ok"] and db.series_meta() == {}


def test_ingest_live_stores_isolates_failures_and_replaces_demo(tmp_db, monkeypatch):
    from app import demo_data
    demo_data.load(seed=1)
    assert db.data_mode() == "demo"
    _fake_sources(monkeypatch)
    summary = ingest.run_live(quiet=True)
    assert {"USDZMW", "DXY", "POLICY_RATE", "CPI_YOY"} <= set(summary["ok"])
    assert any("opper" in f for f in summary["failed"])          # copper failed, others unaffected
    assert db.data_mode() == "live"                              # demo rows removed once real data stored
    assert any(l["status"] == "error" for l in db.recent_ingest_log(20))


def test_total_network_failure_keeps_existing_data(tmp_db, monkeypatch):
    from app import demo_data
    demo_data.load(seed=1)
    for mod, name in ((ingest.frankfurter_boz, "fetch"), (ingest.fred, "fetch"), (ingest.copper, "fetch")):
        monkeypatch.setattr(mod, name, lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(ingest.manual_csv, "fetch", lambda: [])
    summary = ingest.run_live(quiet=True)
    assert summary["ok"] == [] and db.data_mode() == "demo"      # demo data not wiped by a failed live run


def test_second_live_run_cannot_shrink_history(tmp_db, monkeypatch):
    _fake_sources(monkeypatch, fx_rows=400)
    ingest.run_live(quiet=True)
    n = db.count_rows("USDZMW")
    _fake_sources(monkeypatch, fx_rows=60)                       # a truncated response from the provider
    ingest.run_live(quiet=True)
    assert db.count_rows("USDZMW") == n


# ---------------- panel: step series must reach the end of the sample ----------------
def test_policy_rate_steps_are_carried_forward_so_driver_forecast_stays_live(tmp_db):
    from app.config import SERIES
    data = generate(seed=3, start="2014-01-01", end="2026-11-20")
    for sid, s in data.items():
        if sid == "POLICY_RATE":
            s = s[s.diff().fillna(1) != 0]                       # keep only change dates, like the real BoZ file
            s = s.loc[:"2026-03-01"]                             # last change months before the data ends
        m = SERIES[sid]
        db.upsert_series(sid, s, name=m["name"], unit=m["unit"], frequency=m["native"], source="test", is_demo=False)
    panel = build_panel("W-FRI")
    assert panel.levels["POLICY_RATE"].iloc[-1] == panel.levels["POLICY_RATE"].dropna().iloc[-1]
    ctx = ForecastContext(panel)
    assert not np.isnan(ctx.Xv[-1]).any()


# ---------------- scheduler ----------------
def test_refresher_records_state_and_guards_against_overlap(monkeypatch):
    warmed = []
    r = Refresher(interval_hours=1, after=lambda: warmed.append(1))
    monkeypatch.setattr(ingest, "run_live", lambda quiet=True: dict(ok=["USDZMW"], failed=["X"]))
    out = r.run_once()
    assert out["ok"] == ["USDZMW"] and r.state["last_failed"] == ["X"] and warmed == [1] and not r.state["running"]
    monkeypatch.setattr(ingest, "run_live", lambda quiet=True: dict(ok=[], failed=["USDZMW"]))
    r.run_once()
    assert "no series" in r.state["last_error"]
    r._lock.acquire()
    assert r.run_once() is None                                  # overlapping run is skipped
    r._lock.release()


# ---------------- auth ----------------
def test_basic_auth_only_when_password_set(monkeypatch):
    assert client.get("/api/status").status_code == 200
    monkeypatch.setenv("ZTI_PASSWORD", "s3cret")
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/health").status_code == 200
    good = {"Authorization": "Basic " + base64.b64encode(b"treasury:s3cret").decode()}
    bad = {"Authorization": "Basic " + base64.b64encode(b"treasury:nope").decode()}
    assert client.get("/api/status", headers=good).status_code == 200
    assert client.get("/api/status", headers=bad).status_code == 401
    assert client.get("/", headers={"Authorization": "Basic !!!"}).status_code == 401


def test_status_exposes_refresh_state():
    assert "refresh" in client.get("/api/status").json()

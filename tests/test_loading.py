"""USD/ZMW loader fallbacks, file uploader, and the in-app data-loading endpoints."""
import base64
import io
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db, ingest
from app.connectors import frankfurter_boz as fx
from app.connectors.base import FeedError, SeriesResult
from app.main import app, refresher
from app.upload import UploadError, parse_upload

client = TestClient(app)


class FakeResp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


def _records(lo, hi, base="USD"):
    days = pd.bdate_range(lo, hi)
    return [dict(date=d.strftime("%Y-%m-%d"), base=base, quote="ZMW", rate=round(5.0 + 0.001 * i, 4)) for i, d in enumerate(days)]


# ---------------- Frankfurter loader ----------------
def test_daily_history_is_requested_in_one_year_pieces(monkeypatch):
    seen = []

    def fake(url, params=None, timeout=0, retries=3):
        seen.append((params["from"], params["to"]))
        return FakeResp(_records(params["from"], params["to"]))

    monkeypatch.setattr(fx, "http_get", fake)
    s = fx.fetch_daily(start="2013-01-01", end="2015-06-30")
    assert seen == [("2013-01-01", "2013-12-31"), ("2014-01-01", "2014-12-31"), ("2015-01-01", "2015-06-30")]
    assert len(s) == len(pd.bdate_range("2013-01-01", "2015-06-30")) and s.index.is_monotonic_increasing


def test_failed_piece_names_the_dates_and_the_http_error(monkeypatch):
    def fake(url, params=None, timeout=0, retries=3):
        if params["from"].startswith("2014"):
            raise FeedError("https://api.frankfurter.dev/v2/rates -> HTTP 503: deadline expired", status=503)
        return FakeResp(_records(params["from"], params["to"]))

    monkeypatch.setattr(fx, "http_get", fake)
    with pytest.raises(FeedError) as e:
        fx.fetch_daily(start="2013-01-01", end="2015-06-30")
    assert "2014-01-01..2014-12-31" in str(e.value) and "503" in str(e.value)


def test_coverage_check_rejects_stale_late_start_and_gappy_history():
    end = pd.Timestamp("2026-11-20")
    ok = pd.Series(1.0, index=pd.bdate_range("2013-01-02", end))
    fx._check_coverage(ok, "2013-01-01", end, 21)
    with pytest.raises(FeedError, match="starts"):
        fx._check_coverage(ok.loc["2014-06-01":], "2013-01-01", end, 21)
    with pytest.raises(FeedError, match="days old"):
        fx._check_coverage(ok.loc[:"2026-06-01"], "2013-01-01", end, 21)
    gappy = ok.drop(ok.loc["2019-03-01":"2019-05-31"].index)
    with pytest.raises(FeedError, match="gap of"):
        fx._check_coverage(gappy, "2013-01-01", end, 21)


def test_fetch_falls_back_to_weekly_then_blend_and_reports_everything(monkeypatch):
    weekly = pd.Series(18.0, index=pd.date_range("2013-01-07", pd.Timestamp.today().normalize(), freq="W-MON"))
    monkeypatch.setattr(fx, "fetch_daily", lambda: (_ for _ in ()).throw(FeedError("daily 503")))
    monkeypatch.setattr(fx, "fetch_grouped", lambda provider="BOZ": weekly)
    res = fx.fetch()
    assert res[0].series_id == "USDZMW" and "WEEKLY" in res[0].source and "daily 503" in res[0].notes

    def only_blend(provider="BOZ"):
        if provider:
            raise FeedError("weekly 422")
        return weekly

    monkeypatch.setattr(fx, "fetch_grouped", only_blend)
    assert "blended" in fx.fetch()[0].source

    monkeypatch.setattr(fx, "fetch_grouped", lambda provider="BOZ": (_ for _ in ()).throw(FeedError("nope")))
    with pytest.raises(RuntimeError) as e:
        fx.fetch()
    assert all(tag in str(e.value) for tag in ("[daily]", "[weekly]", "[blend]"))


def test_parse_rates_ignores_other_pairs_and_duplicates():
    recs = [dict(date="2026-11-20", base="USD", quote="ZMW", rate=20.4), dict(date="2026-11-20", base="USD", quote="ZMW", rate=20.5),
            dict(date="2026-11-20", base="USD", quote="USD", rate=1.0), dict(date="2026-11-20", base="EUR", quote="ZMW", rate=23.0)]
    s = fx.parse_rates(recs)
    assert len(s) == 1 and s.iloc[0] == 20.5


def test_ingest_clears_source_error_after_a_later_success(tmp_db, monkeypatch):
    good = pd.Series(18.0, index=pd.bdate_range("2013-01-02", pd.Timestamp.today().normalize()))
    monkeypatch.setattr(ingest.frankfurter_boz, "fetch", lambda: (_ for _ in ()).throw(FeedError("HTTP 503")))
    monkeypatch.setattr(ingest.fred, "fetch", lambda: [])
    monkeypatch.setattr(ingest.copper, "fetch", lambda: [])
    ingest.run_live(quiet=True)
    assert any("Frankfurter" in p["series_id"] and "503" in p["message"] for p in db.latest_problems())
    monkeypatch.setattr(ingest.frankfurter_boz, "fetch", lambda: [SeriesResult("USDZMW", good, "fake")])
    ingest.run_live(quiet=True)
    assert not any("Frankfurter" in p["series_id"] for p in db.latest_problems())


# ---------------- file parser ----------------
BOZ_STYLE = ("Bank of Zambia\nAverage exchange rates\n\nDate,Buying,Selling,Mid\n"
             + "\n".join(f"{d:02d}/01/2013,5.{d:02d},5.{d + 2:02d},5.{d + 1:02d}" for d in range(14, 28)) + "\n")


def test_parse_boz_style_csv_with_title_rows_and_dayfirst_dates():
    s, info = parse_upload(BOZ_STYLE.encode(), "rates.csv")
    assert info["date_column"] == "Date" and info["value_column"] == "Mid" and info["rows"] == 14
    assert s.index[0] == pd.Timestamp("2013-01-14") and s.iloc[0] == pytest.approx(5.15)
    assert set(info["numeric_columns"]) == {"Buying", "Selling", "Mid"}
    s2, info2 = parse_upload(BOZ_STYLE.encode(), "rates.csv", value_col="selling")
    assert info2["value_column"] == "Selling" and s2.iloc[0] == pytest.approx(5.16)
    with pytest.raises(UploadError, match="not found"):
        parse_upload(BOZ_STYLE.encode(), "rates.csv", value_col="nonsense")


def test_parse_iso_csv_thousands_separators_and_excel():
    csv = 'date,value\n' + "\n".join(f'2025-01-{d:02d},"8,5{d:02d}.50"' for d in range(1, 11)) + "\n"
    s, _ = parse_upload(csv.encode(), "copper.csv")
    assert len(s) == 10 and s.iloc[0] == pytest.approx(8501.5)

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Zambia exchange rates", None]); ws.append([None, None]); ws.append(["Date", "ZMW per USD"])
    for i in range(12):
        ws.append([pd.Timestamp("2024-03-01") + pd.Timedelta(days=i), 26.0 + i / 10])
    buf = io.BytesIO(); wb.save(buf)
    s, info = parse_upload(buf.getvalue(), "book.xlsx")
    assert len(s) == 12 and info["value_column"] == "ZMW per USD" and s.index[0] == pd.Timestamp("2024-03-01")


@pytest.mark.parametrize("content,name,msg", [
    (b"", "a.csv", "empty"),
    (b"just,words\nhello,world\nfoo,bar\n", "a.csv", "date column"),
    (b"date,label\n2025-01-01,a\n2025-01-02,b\n2025-01-03,c\n", "a.csv", "numeric"),
    (b"onlyonecolumn\n1\n2\n", "a.csv", "two columns"),
    (b"x" * 5_000_001, "big.csv", "5 MB"),
])
def test_parse_rejects_bad_files_with_clear_messages(content, name, msg):
    with pytest.raises(UploadError, match=msg):
        parse_upload(content, name)


# ---------------- endpoints ----------------
def _csv(start, n, base=15.0):
    return ("Date,Mid\n" + "\n".join(f"{(pd.Timestamp(start) + pd.Timedelta(days=i)).date()},{base + i / 100:.2f}" for i in range(n))).encode()


def test_upload_stores_merges_and_replaces(tmp_db):
    r = client.post("/api/upload", params=dict(series_id="USDZMW", filename="a.csv"), content=_csv("2020-01-01", 60))
    assert r.status_code == 200 and r.json()["stored_rows"] == 60
    assert db.series_meta()["USDZMW"]["source"] == "upload: a.csv" and db.data_mode() == "live"
    r = client.post("/api/upload", params=dict(series_id="USDZMW", filename="b.csv"), content=_csv("2020-02-15", 60))
    assert r.json()["stored_rows"] == 60 + 60 - 15        # Feb 15-29 overlap is overwritten, not doubled
    r = client.post("/api/upload", params=dict(series_id="USDZMW", filename="c.csv", replace="true"), content=_csv("2021-01-01", 30))
    assert r.json()["stored_rows"] == 30 and db.load_series("USDZMW").index[0] == pd.Timestamp("2021-01-01")
    assert db.count_rows("USDZMW") == 30
    assert client.get("/api/overview").status_code == 200   # the app now works with an uploaded exchange rate


def test_upload_rejects_unknown_series_bad_files_and_implausible_values(tmp_db):
    assert client.post("/api/upload", params=dict(series_id="NOPE"), content=_csv("2020-01-01", 20)).status_code == 404
    bad = client.post("/api/upload", params=dict(series_id="USDZMW", filename="x.csv"), content=b"hello world")
    assert bad.status_code == 422 and bad.json()["detail"]
    wrong_units = client.post("/api/upload", params=dict(series_id="USDZMW", filename="x.csv"), content=_csv("2020-01-01", 40, base=500))
    assert wrong_units.status_code == 422 and "plausible range" in wrong_units.json()["detail"]
    assert db.series_meta() == {}                            # nothing was stored


def test_upload_replaces_demo_data_only_when_real_data_arrives(tmp_db):
    from app import demo_data
    demo_data.load(seed=2)
    assert db.data_mode() == "demo"
    client.post("/api/upload", params=dict(series_id="USDZMW", filename="a.csv"), content=_csv("2020-01-01", 60))
    assert db.data_mode() == "live" and set(db.series_meta()) == {"USDZMW"}


def test_public_host_without_password_cannot_change_data(tmp_db, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(refresher, "run_once", lambda: None)      # no network in tests
    assert client.post("/api/upload", params=dict(series_id="USDZMW"), content=_csv("2020-01-01", 20)).status_code == 403
    assert client.post("/api/refresh").status_code == 200          # retry is allowed: it only fetches public data
    assert client.get("/api/status").json()["controls_locked"] is True
    monkeypatch.setenv("ZTI_PASSWORD", "pw")
    hdr = {"Authorization": "Basic " + base64.b64encode(b"treasury:pw").decode()}
    ok = client.post("/api/upload", params=dict(series_id="USDZMW", filename="a.csv"), content=_csv("2020-01-01", 60), headers=hdr)
    assert ok.status_code == 200
    assert client.post("/api/upload", params=dict(series_id="USDZMW"), content=_csv("2020-01-01", 20)).status_code == 401


def test_refresh_endpoint_starts_background_run_once(tmp_db, monkeypatch):
    done = threading.Event()
    monkeypatch.setattr(refresher, "run_once", lambda: done.set())
    assert client.post("/api/refresh").json() == dict(started=True, running=False)
    assert done.wait(2)
    monkeypatch.setitem(refresher.state, "running", True)
    assert client.post("/api/refresh").json() == dict(started=False, running=True)


def test_status_lists_problems_registry_and_lock_state(tmp_db):
    db.log_ingest("Bank of Zambia FX (via Frankfurter)", 0, "error", "FeedError: HTTP 503 deadline expired")
    db.log_ingest("COPPER_USD", 5000, "warn", "2 unusually large moves")
    db.log_ingest("DXY", 3000, "ok", "")
    st = client.get("/api/status").json()
    assert {p["series_id"] for p in st["problems"]} == {"Bank of Zambia FX (via Frankfurter)", "COPPER_USD"}
    assert any(r["series_id"] == "USDZMW" for r in st["registry"]) and st["controls_locked"] is False
    db.log_ingest("Bank of Zambia FX (via Frankfurter)", 1, "ok", "")
    assert "Bank of Zambia FX (via Frankfurter)" not in {p["series_id"] for p in client.get("/api/status").json()["problems"]}


def test_multi_currency_tab_file_prefers_usd_and_ignores_footer():
    rows = ["Bank of Zambia - Average Exchange Rates (ZMW per unit)", "", "Date\tEUR\tGBP\tUSD\tZAR"]
    rows += [f"{d:02d}-Mar-2024\t{28 + d / 10:.2f}\t{33 + d / 10:.2f}\t{26 + d / 10:.2f}\t1.{40 + d}" for d in range(1, 21)]
    rows.append("Source: Bank of Zambia")
    s, info = parse_upload("\n".join(rows).encode(), "rates.txt", prefer=("usd",))
    assert info["value_column"] == "USD" and info["rows"] == 20 and s.iloc[0] == pytest.approx(26.1)
    assert info["first"] == "2024-03-01" and set(info["numeric_columns"]) == {"EUR", "GBP", "USD", "ZAR"}


def test_month_first_dates_detected_and_ambiguous_files_default_to_day_first():
    us = "date;rate\n" + "\n".join(f"03/{d:02d}/2024;{20 + d / 10:.2f}" for d in range(13, 28))
    assert parse_upload(us.encode(), "us.csv")[1]["first"] == "2024-03-13"
    amb = "Date,Rate\n" + "\n".join(f"{d:02d}/04/2024,{20 + d / 10:.2f}" for d in range(1, 10))
    assert parse_upload(amb.encode(), "amb.csv")[1]["first"] == "2024-04-01"


def test_upload_endpoint_picks_the_dollar_column_for_usdzmw(tmp_db):
    rows = ["Date,EUR,USD"] + [f"2024-03-{d:02d},{28 + d / 10:.2f},{26 + d / 10:.2f}" for d in range(1, 21)]
    r = client.post("/api/upload", params=dict(series_id="USDZMW", filename="r.csv"), content="\n".join(rows).encode())
    assert r.status_code == 200 and r.json()["upload"]["value_column"] == "USD"
    assert db.load_series("USDZMW").iloc[0] == pytest.approx(26.1)


def test_manual_file_is_only_a_fallback_for_feed_backed_series(tmp_db, monkeypatch, tmp_path):
    from app.connectors import manual_csv
    folder = tmp_path / "manual"
    folder.mkdir()
    (folder / "USDZMW.csv").write_text("# source: my file\ndate,value\n" + "\n".join(
        f"{d.date()},12.34" for d in pd.bdate_range("2020-01-01", periods=60)))
    monkeypatch.setattr(manual_csv, "MANUAL_DIR", folder)
    monkeypatch.setattr(ingest.fred, "fetch", lambda: [])
    monkeypatch.setattr(ingest.copper, "fetch", lambda: [])

    monkeypatch.setattr(ingest.frankfurter_boz, "fetch", lambda: (_ for _ in ()).throw(FeedError("HTTP 503")))
    ingest.run_live(quiet=True)
    assert db.series_meta()["USDZMW"]["source"].startswith("manual: my file")      # feed down -> file used

    live = pd.Series(20.0, index=pd.bdate_range("2013-01-02", pd.Timestamp.today().normalize()))
    monkeypatch.setattr(ingest.frankfurter_boz, "fetch", lambda: [SeriesResult("USDZMW", live, "fake live feed")])
    ingest.run_live(quiet=True)
    assert db.series_meta()["USDZMW"]["source"] == "fake live feed"                 # feed back -> feed wins


def test_demo_endpoint_on_empty_db_then_refused_when_real_fx_exists(tmp_db):
    assert client.get("/api/status").json()["mode"] == "empty"
    assert client.post("/api/demo").status_code == 200
    assert client.get("/api/status").json()["mode"] == "demo"
    assert client.get("/api/overview").status_code == 200
    db.clear_demo()
    ok = client.post("/api/upload", params=dict(series_id="USDZMW", filename="a.csv"), content=_csv("2020-01-01", 60))
    assert ok.status_code == 200
    assert client.post("/api/demo").status_code == 409


def test_auto_ingest_defaults(monkeypatch):
    from app.main import _auto_ingest_enabled
    monkeypatch.delenv("ZTI_AUTO_INGEST", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    assert _auto_ingest_enabled() is False
    monkeypatch.setenv("RENDER", "true")
    assert _auto_ingest_enabled() is True
    monkeypatch.setenv("ZTI_AUTO_INGEST", "0")
    assert _auto_ingest_enabled() is False


def test_seed_files_load_offline_and_failed_live_run_keeps_demo(tmp_db, monkeypatch):
    from app import ingest, demo_data
    out = ingest.load_seed_files()
    assert "POLICY_RATE" in out["ok"] and out["events"] > 0
    assert db.data_mode() == "live"          # no USDZMW, but real rows exist
    db.clear_demo()
    demo_data.load()
    def boom(): raise RuntimeError("network down")
    monkeypatch.setattr(ingest.frankfurter_boz, "fetch", boom)
    monkeypatch.setattr(ingest.fred, "fetch", boom)
    monkeypatch.setattr(ingest.copper, "fetch", boom)
    res = ingest.run_live(quiet=True)
    assert "USDZMW" in db.series_meta() and db.series_meta()["USDZMW"]["is_demo"]
    assert "USDZMW" in res["failed"] or "Bank of Zambia FX (via Frankfurter)" in res["failed"]

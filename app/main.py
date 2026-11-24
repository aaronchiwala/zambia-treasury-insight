"""FastAPI app: JSON API + static dashboard.   Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

import base64
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, events_derived, ingest, upload as upload_mod
from .analytics import attribution, backtest, forecast, history, scenario
from .analytics.panel import DataError, build_panel
from .config import GROUP_LABELS, SERIES, TARGET
from .scheduler import Refresher
from .validate import summarize, validate_series

STATIC = Path(__file__).resolve().parent.parent / "static"
db.init()

_cache: dict = {}


def _warm_cache() -> None:
    """After a data refresh, precompute the (slow) backtest so the first Forecast click is instant."""
    _backtest("W-FRI", (1, 4, 8, 12), 2)


refresher = Refresher(after=_warm_cache)


def _auto_ingest_enabled() -> bool:
    """On by default on Render (which sets RENDER); ZTI_AUTO_INGEST=1/0 overrides either way."""
    v = os.environ.get("ZTI_AUTO_INGEST")
    if v is not None:
        return v == "1"
    return bool(os.environ.get("RENDER"))


def _load_seeds_if_empty() -> None:
    """Offline start-up: policy rate, inflation and events come from files in the repo, so the DB is never blank."""
    try:
        if db.data_mode() == "empty":
            ingest.load_seed_files()
    except Exception as e:  # noqa: BLE001 - never block start-up
        db.log_ingest("seed files", 0, "error", f"{type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_seeds_if_empty()
    if _auto_ingest_enabled():
        refresher.start()
    yield
    refresher.stop()


app = FastAPI(title="Zambia Treasury Insight", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def optional_basic_auth(request, call_next):
    """If ZTI_PASSWORD is set, every route except /api/health needs HTTP Basic credentials."""
    password = os.environ.get("ZTI_PASSWORD")
    if not password or request.url.path == "/api/health":
        return await call_next(request)
    user = os.environ.get("ZTI_USER", "treasury")
    ok = False
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            u, _, p = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            ok = secrets.compare_digest(u.encode(), user.encode()) and secrets.compare_digest(p.encode(), password.encode())
        except Exception:  # noqa: BLE001 - malformed header means not authenticated
            ok = False
    if ok:
        return await call_next(request)
    return Response("Authentication required", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Zambia Treasury Insight"'})


def cached(fn):
    """Memoise by (function, args) and invalidate whenever the database file changes."""
    @wraps(fn)
    def inner(*args):
        key = (fn.__name__, args, db.db_stamp())
        if key not in _cache:
            if len(_cache) > 200:
                _cache.clear()
            _cache[key] = fn(*args)
        return _cache[key]
    return inner


def _events_for_mode() -> list[dict]:
    mode = db.data_mode()
    events = db.load_events(is_demo=(mode == "demo"))
    if mode != "demo":
        meta = db.series_meta().get("POLICY_RATE")
        if meta and not meta["is_demo"]:
            events = events_derived.merge(events, events_derived.from_policy_rate(db.load_series("POLICY_RATE")))
    return events


def _panel(freq: str):
    try:
        return build_panel(freq)
    except DataError as e:
        raise HTTPException(409, str(e))


@cached
def _backtest(freq: str, horizons: tuple, step: int):
    return backtest.run_backtest(_panel(freq), horizons=horizons, step=step)


@app.get("/api/status")
def status():
    meta = db.series_meta()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    series = []
    for sid, m in meta.items():
        last = pd.Timestamp(m["last_date"]) if m["last_date"] else None
        age = int((today - last).days) if last is not None else None
        reg = SERIES.get(sid, {})
        stale_limit = reg.get("stale_days") or {"D": 7, "E": 200, "W": 14, "M": 75, "Q": 130}.get(reg.get("native", "D"), 14)
        series.append(dict(series_id=sid, name=m["name"], unit=m["unit"], native=SERIES.get(sid, {}).get("native"),
                           source=m["source"], is_demo=bool(m["is_demo"]), n_obs=m["n_obs"], first_date=m["first_date"],
                           last_date=m["last_date"], age_days=age, stale=bool(age is not None and age > stale_limit),
                           last_fetched=m["last_fetched"]))
    mode = db.data_mode()
    notes = []
    if mode in ("demo", "mixed"):
        notes.append("Synthetic demo data is loaded. Values are NOT real market data.")
    if TARGET not in meta:
        notes.append("No USD/ZMW series loaded.")
    if refresher.state.get("last_error"):
        notes.append("Last automatic refresh: " + str(refresher.state["last_error"]))
    return dict(mode=mode, series=sorted(series, key=lambda s: s["series_id"]), notes=notes,
                refresh=dict(refresher.state), problems=db.latest_problems(),
                registry=[dict(series_id=k, name=v["name"], unit=v["unit"], native=v["native"]) for k, v in SERIES.items()],
                controls_locked=_hosted_unprotected(), ingest_log=db.recent_ingest_log(12),
                server_time=datetime.now(timezone.utc).isoformat(timespec="seconds"))


@app.get("/api/health")
def health():
    return dict(ok=True)


def _hosted_unprotected() -> bool:
    """On a public host (Render sets RENDER) without a password, nobody may change data from the browser."""
    return bool(os.environ.get("RENDER")) and not os.environ.get("ZTI_PASSWORD")


def _require_controls() -> None:
    if _hosted_unprotected():
        raise HTTPException(403, "Data controls are disabled on a public host until ZTI_PASSWORD is set "
                                 "(Render dashboard > Environment). Set it, redeploy, then sign in.")


@app.post("/api/refresh")
def refresh_now():
    """Re-run the live data load in the background (same as the automatic refresh)."""
    if refresher.state.get("running"):
        return dict(started=False, running=True)
    threading.Thread(target=refresher.run_once, name="zti-refresh-now", daemon=True).start()
    return dict(started=True, running=False)


@app.post("/api/demo")
def load_demo():
    """One-click synthetic data so the app can be explored. Refused once real USD/ZMW data exists."""
    meta = db.series_meta().get(TARGET)
    if meta and not meta["is_demo"]:
        raise HTTPException(409, "Real USD/ZMW data is already loaded; demo data would only confuse it.")
    from . import demo_data
    out = demo_data.load()
    _cache.clear()
    return dict(ok=True, detail=str(out))


@app.post("/api/upload")
async def upload_series(request: Request, series_id: str, filename: str = "", value_col: str | None = None,
                        replace: bool = False):
    """Load a CSV/XLSX the user supplies (e.g. a Bank of Zambia rates download) into one series.

    By default the rows are MERGED into any stored real data (same dates are overwritten); replace=true
    discards what was there. The data passes the same quality gate as automatic feeds."""
    _require_controls()
    if series_id not in SERIES:
        raise HTTPException(404, f"Unknown series {series_id!r}. Known: {', '.join(SERIES)}")
    try:
        hint = ("usd",) if series_id == "USDZMW" else ()      # multi-currency tables: take the dollar column
        new, info = upload_mod.parse_upload(await request.body(), filename, value_col, prefer=hint)
    except upload_mod.UploadError as e:
        raise HTTPException(422, str(e))
    meta = db.series_meta().get(series_id)
    if meta and not meta["is_demo"] and not replace:
        old = db.load_series(series_id)
        new = pd.concat([old, new])
        new = new[~new.index.duplicated(keep="last")].sort_index()
    cleaned, issues = validate_series(series_id, new)
    status, msg = summarize(issues)
    if cleaned is None:
        raise HTTPException(422, msg or "data rejected by the quality checks")
    m = SERIES[series_id]
    db.clear_demo()
    n = db.upsert_series(series_id, cleaned, name=m["name"], unit=m["unit"], frequency=m["native"],
                         source=f"upload: {filename or 'file'}", is_demo=False,
                         notes=f"Uploaded by a user; value column {info['value_column']!r}.")
    db.log_ingest(series_id, n, status, msg or f"uploaded {info['rows']} rows from {filename or 'file'}")
    if not db.load_events(is_demo=False):
        ingest.load_events()
    _cache.clear()
    return dict(ok=True, series_id=series_id, stored_rows=n, upload=info, quality=[dict(level=a, message=b) for a, b in issues])


@app.get("/api/overview")
def overview():
    meta = db.series_meta()
    if TARGET not in meta:
        raise HTTPException(409, "No data loaded. Run `python -m app.ingest --demo` or `--live`.")
    cards = []
    for sid, m in meta.items():
        s = db.load_series(sid)
        if s.empty:
            continue
        last = float(s.iloc[-1])
        def chg(days):
            ref = s.loc[:s.index[-1] - pd.Timedelta(days=days)]
            if ref.empty:
                return None
            r = float(ref.iloc[-1])
            if SERIES[sid]["kind"] == "rate":
                return last - r
            return (last / r - 1) * 100 if r else None
        cards.append(dict(series_id=sid, name=m["name"], unit=m["unit"], kind=SERIES[sid]["kind"], last=last,
                          date=s.index[-1].strftime("%Y-%m-%d"), chg_1w=chg(7), chg_4w=chg(28), chg_12w=chg(84), chg_52w=chg(364),
                          is_demo=bool(m["is_demo"])))
    return dict(cards=sorted(cards, key=lambda c: (c["series_id"] != TARGET, c["series_id"])), mode=db.data_mode())


@app.get("/api/series")
def series(ids: str = Query(TARGET), start: str | None = None, end: str | None = None, max_points: int = 2500):
    out = {}
    for sid in [i.strip() for i in ids.split(",") if i.strip()]:
        s = db.load_series(sid)
        if s.empty:
            continue
        if start:
            s = s.loc[start:]
        if end:
            s = s.loc[:end]
        if len(s) > max_points:  # decimate for the browser, keep the last point
            s = pd.concat([s.iloc[:: int(np.ceil(len(s) / max_points))], s.iloc[[-1]]]).loc[lambda x: ~x.index.duplicated()]
        out[sid] = dict(dates=[d.strftime("%Y-%m-%d") for d in s.index], values=[round(float(v), 6) for v in s.values],
                        name=SERIES.get(sid, {}).get("name", sid), unit=SERIES.get(sid, {}).get("unit", ""))
    return out


@app.get("/api/events")
def events():
    return dict(events=_events_for_mode(), mode=db.data_mode())


@app.get("/api/explain")
def explain(start: str, end: str, freq: str = "W-FRI", window: int | None = None):
    p = _panel(freq)
    try:
        fit = attribution.fit_driver_model(p, window=window)
        res = attribution.explain_period(p, fit, start, end, _events_for_mode())
    except ValueError as e:
        raise HTTPException(422, str(e))
    res["shocks"] = attribution.shock_periods(p, fit, _events_for_mode())[-12:]
    res["skipped"] = p.skipped
    return res


@app.get("/api/history")
def history_drivers(threshold: float = Query(8.0, ge=3.0, le=30.0), freq: str = "W-FRI"):
    """Major swings in USD/ZMW over the whole history, each attributed to drivers."""
    p = _panel(freq)
    try:
        fit = attribution.fit_driver_model(p)
        return history.historical_drivers(p, fit, _events_for_mode(), threshold / 100)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/model")
def model_summary(freq: str = "W-FRI", window: int | None = None):
    p = _panel(freq)
    fit = attribution.fit_driver_model(p, window=window)
    rb = attribution.rolling_betas(p, fit)
    return dict(
        freq=freq, r2=fit.r2, n=fit.n, sample=[fit.sample_start.strftime("%Y-%m-%d"), fit.sample_end.strftime("%Y-%m-%d")],
        drivers=[dict(driver=d, label=GROUP_LABELS.get(d, d), beta=fit.beta(d), se=float(fit.se[d]),
                      p_value=float(fit.pvalues[d]), series=p.used[d]) for d in fit.drivers],
        excluded=fit.excluded, skipped=p.skipped, rolling=rb)


@app.get("/api/backtest")
def get_backtest(freq: str = "W-FRI", horizons: str = "1,4,8,12", step: int = 2):
    hs = tuple(int(x) for x in horizons.split(",") if x.strip())
    return _backtest(freq, hs, step)


@app.get("/api/forecast")
def get_forecast(h: int = Query(12, ge=1, le=26), model: str = "auto", freq: str = "W-FRI"):
    p = _panel(freq)
    hs = tuple(sorted({1, 4, 8, 12} | {min(h, 12)}))
    bt = _backtest(freq, hs, 2)
    try:
        f = forecast.make_forecast(p, h, model, bt["summary"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    f["backtest_summary"] = [r for r in bt["summary"]]
    f["is_demo"] = p.is_demo
    return f


class ScenarioIn(BaseModel):
    shocks: dict[str, float] = Field(..., description="driver -> shock. Prices in % change, rates in basis points.")
    horizon: int = Field(12, ge=1, le=52)
    window: int = Field(156, ge=60, le=1000)
    freq: str = "W-FRI"


@app.post("/api/scenario")
def post_scenario(body: ScenarioIn):
    p = _panel(body.freq)
    try:
        return scenario.run_scenario(p, body.shocks, body.horizon, body.window)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/scenario/presets")
def presets(freq: str = "W-FRI"):
    p = _panel(freq)
    return dict(presets=scenario.available_presets(p),
                drivers=[dict(driver=d, label=GROUP_LABELS.get(d, d), kind=p.kinds[d]) for d in p.drivers])


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

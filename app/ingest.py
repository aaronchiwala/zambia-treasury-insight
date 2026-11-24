"""Data ingestion.

    python -m app.ingest --check    # dry run: fetch everything, validate, print a report, write NOTHING
    python -m app.ingest --live     # BoZ FX + FRED + copper + data/manual/*.csv + data/events.csv
    python -m app.ingest --manual   # only re-read data/manual/*.csv and events
    python -m app.ingest --demo     # synthetic data so the app runs end-to-end (clearly labelled)

Each source runs in isolation: one broken feed never blocks the others, and every series passes the quality
gate in app/validate.py before it is stored. Demo rows are removed only once real data has been stored, so a
total network failure never leaves the database empty.
"""
from __future__ import annotations

import argparse
import csv

from . import db, demo_data
from .config import EVENTS_CSV, SERIES
from .connectors import copper, fred, frankfurter_boz, manual_csv
from .validate import summarize, validate_series


class Run:
    """State for one ingest pass."""

    def __init__(self, dry: bool = False, quiet: bool = False):
        self.dry, self.quiet = dry, quiet
        self.ok: list[str] = []
        self.failed: list[str] = []
        self.rows: list[dict] = []
        self._demo_cleared = False
        # Demo rows are replaced only when REAL USD/ZMW has loaded in this run, so a failed refresh never wipes them.
        self.fx_ok = False

    def say(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def store(self, res) -> bool:
        sid = res.series_id
        if res.data.empty:
            self._fail(sid, res.notes or "no data")
            return False
        cleaned, issues = validate_series(sid, res.data, existing_rows=0 if self.dry else db.count_rows(sid))
        status, msg = summarize(issues)
        if cleaned is None:
            self._fail(sid, msg)
            return False
        m = SERIES[sid]
        if not self.dry:
            if sid == "USDZMW":
                self.fx_ok = True
            meta = db.series_meta()
            demo_present = any(v["is_demo"] for v in meta.values())
            if demo_present and not self.fx_ok and not self._demo_cleared:
                self._fail(sid, "kept the demo data: real USD/ZMW has not loaded in this run")
                return False
            if not self._demo_cleared:
                db.clear_demo()
                self._demo_cleared = True
            n = db.upsert_series(sid, cleaned, name=m["name"], unit=m["unit"], frequency=m["native"],
                                 source=res.source, is_demo=False, notes=res.notes)
            db.log_ingest(sid, n, status, msg)
        else:
            n = len(cleaned)
        self.ok.append(sid)
        self.rows.append(dict(series=sid, rows=n, first=str(cleaned.index.min().date()),
                              last=str(cleaned.index.max().date()), status=status, note=msg, source=res.source))
        flag = "✓" if status == "ok" else "!"
        self.say(f"  {flag} {sid:14s} {n:6d} rows  {cleaned.index.min().date()} → {cleaned.index.max().date()}"
                 f"  [{res.source[:70]}]" + (f"\n      note: {msg}" if msg else ""))
        return True

    def _fail(self, sid: str, why: str) -> None:
        self.failed.append(sid)
        self.rows.append(dict(series=sid, rows=0, first="", last="", status="error", note=why, source=""))
        if not self.dry:
            db.log_ingest(sid, 0, "error", why)
        self.say(f"  ✗ {sid:14s} {why[:150]}")

    def source(self, name: str, fn) -> None:
        self.say(f"{name}:")
        try:
            results = fn()
        except Exception as e:  # noqa: BLE001 - isolate each source
            why = f"{type(e).__name__}: {e}"
            if not self.dry:
                db.log_ingest(name, 0, "error", why)
            self.failed.append(name)
            self.rows.append(dict(series=name, rows=0, first="", last="", status="error", note=why, source=""))
            self.say(f"  ✗ failed: {why[:300]}")
            return
        stored = sum(1 for r in results if self.store(r))
        if stored and not self.dry:
            db.log_ingest(name, stored, "ok", "")      # clears an earlier source-level error from the problems list


def load_events() -> int:
    if not EVENTS_CSV.exists():
        return 0
    with EVENTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return db.replace_events(rows, is_demo=False)


def load_seed_files() -> dict:
    """Load only the files shipped in the repo (no network): manual CSVs and curated events."""
    db.init()
    run = Run(quiet=True)
    run.source("Seed files", manual_csv.fetch)
    return dict(ok=run.ok, failed=run.failed, events=load_events())


def run_live(dry: bool = False, quiet: bool = False) -> dict:
    """Pull every live source plus the manual CSVs. Safe to call repeatedly (idempotent)."""
    if not dry:
        db.init()
    run = Run(dry=dry, quiet=quiet)
    run.source("Bank of Zambia FX (via Frankfurter)", frankfurter_boz.fetch)
    run.source("FRED global drivers", fred.fetch)
    run.source("Copper daily (Yahoo, then Stooq)", copper.fetch)
    live_ok = set(run.ok)

    def manual() -> list:
        """Manual files fill series that have no feed (policy rate, inflation) and act as a FALLBACK for feed-backed
        series. They never override a live feed that just loaded, so a stale file cannot hide fresh data."""
        kept = []
        for r in manual_csv.fetch():
            if r.series_id in live_ok:
                run.say(f"  - {r.series_id}: live feed loaded, so the manual file is not used")
            else:
                kept.append(r)
        return kept

    run.source("Manual CSVs (policy rate, inflation; fallback for failed feeds)", manual)
    events = 0
    if not dry:
        events = load_events()
    run.say(f"events loaded: {events}" if not dry else "events: skipped (dry run)")
    return dict(ok=run.ok, failed=run.failed, rows=run.rows, events=events, dry=dry)


def print_report(summary: dict) -> None:
    print("\n=== Summary ===")
    print(f"stored OK ({len(summary['ok'])}): {', '.join(summary['ok']) or '-'}")
    print(f"FAILED   ({len(summary['failed'])}): {', '.join(summary['failed']) or '-'}")
    core = {"USDZMW", "COPPER_USD", "DXY"}
    missing = sorted(core - set(summary["ok"]))
    if missing:
        print(f"\nCore series missing: {', '.join(missing)}. The app will run, but with reduced explanatory power.")
        if "USDZMW" in missing:
            print("USD/ZMW is required. Check the network/proxy, or put your own USDZMW.csv in data/manual/.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--check", action="store_true", help="dry run: fetch and validate, write nothing")
    a = ap.parse_args()
    if a.demo:
        db.init()
        print(demo_data.load())
        return
    if a.check:
        print_report(run_live(dry=True))
        return
    if a.live:
        print_report(run_live())
        return
    if a.manual:
        db.init()
        run = Run()
        run.source("Manual CSVs", manual_csv.fetch)
        print(f"events loaded: {load_events()}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()

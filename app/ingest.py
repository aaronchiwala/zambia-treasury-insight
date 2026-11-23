"""Data ingestion CLI.

    python -m app.ingest --demo     # synthetic data so the app runs end-to-end
    python -m app.ingest --live     # BoZ FX + FRED + copper + data/manual/*.csv + data/events.csv
    python -m app.ingest --manual   # only re-read data/manual/*.csv and events

Each connector failure is logged and skipped; one broken feed never blocks the others.
Live ingest first removes demo rows so synthetic and real data are never mixed.
"""
from __future__ import annotations

import argparse
import csv

from . import db, demo_data
from .config import EVENTS_CSV, SERIES
from .connectors import fred, frankfurter_boz, manual_csv, yahoo


def _store(res) -> bool:
    sid = res.series_id
    if res.data.empty:
        db.log_ingest(sid, 0, "error", res.notes)
        print(f"  ✗ {sid:14s} no data ({res.notes[:80]})")
        return False
    m = SERIES[sid]
    n = db.upsert_series(sid, res.data, name=m["name"], unit=m["unit"], frequency=m["native"],
                         source=res.source, is_demo=False, notes=res.notes)
    db.log_ingest(sid, n, "ok")
    print(f"  ✓ {sid:14s} {n:6d} rows  {res.data.index.min().date()} → {res.data.index.max().date()}  [{res.source}]")
    return True


def _run(name: str, fn) -> int:
    print(f"{name}:")
    try:
        results = fn()
    except Exception as e:
        db.log_ingest(name, 0, "error", repr(e))
        print(f"  ✗ {name} failed: {e!r}")
        return 0
    return sum(_store(r) for r in results)


def load_events() -> int:
    if not EVENTS_CSV.exists():
        return 0
    with EVENTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return db.replace_events(rows, is_demo=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--manual", action="store_true")
    a = ap.parse_args()
    db.init()
    if a.demo:
        print(demo_data.load())
        return
    if a.live:
        db.clear_demo()
        _run("Bank of Zambia FX (Frankfurter)", frankfurter_boz.fetch)
        _run("FRED global drivers", fred.fetch)
        _run("Copper (Yahoo, best effort)", yahoo.fetch)
    if a.live or a.manual:
        _run("Manual CSVs", manual_csv.fetch)
        print(f"events loaded: {load_events()}")
    if not (a.demo or a.live or a.manual):
        ap.print_help()


if __name__ == "__main__":
    main()

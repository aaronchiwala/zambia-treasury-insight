"""SQLite storage. Swap for PostgreSQL/TimescaleDB later by re-implementing this module only."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS series_meta(
  series_id TEXT PRIMARY KEY, name TEXT, unit TEXT, frequency TEXT,
  source TEXT, is_demo INTEGER NOT NULL DEFAULT 0, notes TEXT, last_fetched TEXT
);
CREATE TABLE IF NOT EXISTS observations(
  series_id TEXT NOT NULL, date TEXT NOT NULL, value REAL NOT NULL,
  PRIMARY KEY(series_id, date)
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, title TEXT NOT NULL,
  category TEXT, description TEXT, source TEXT, verified INTEGER DEFAULT 0,
  is_demo INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ingest_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, series_id TEXT, run_at TEXT,
  rows INTEGER, status TEXT, message TEXT
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")  # background refresh can write while the API reads
        yield con
        con.commit()
    finally:
        con.close()


def init() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_series(series_id: str, data: pd.Series, *, name: str, unit: str, frequency: str,
                  source: str, is_demo: bool, notes: str = "") -> int:
    """Replace a series wholesale (idempotent re-ingest)."""
    s = data.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    rows = [(series_id, d.strftime("%Y-%m-%d"), float(v)) for d, v in s.items()]
    with connect() as con:
        con.execute("DELETE FROM observations WHERE series_id=?", (series_id,))
        con.executemany("INSERT INTO observations VALUES (?,?,?)", rows)
        con.execute(
            "INSERT OR REPLACE INTO series_meta VALUES (?,?,?,?,?,?,?,?)",
            (series_id, name, unit, frequency, source, int(is_demo), notes, _now()),
        )
    return len(rows)


def load_series(series_id: str) -> pd.Series:
    with connect() as con:
        df = pd.read_sql_query(
            "SELECT date, value FROM observations WHERE series_id=? ORDER BY date", con, params=(series_id,)
        )
    if df.empty:
        return pd.Series(dtype=float, name=series_id)
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]), name=series_id)
    return s


def series_meta() -> dict[str, dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT m.*, (SELECT COUNT(*) FROM observations o WHERE o.series_id=m.series_id) AS n_obs,"
            " (SELECT MIN(date) FROM observations o WHERE o.series_id=m.series_id) AS first_date,"
            " (SELECT MAX(date) FROM observations o WHERE o.series_id=m.series_id) AS last_date"
            " FROM series_meta m"
        ).fetchall()
    return {r["series_id"]: dict(r) for r in rows}


def replace_events(rows: list[dict], *, is_demo: bool) -> int:
    with connect() as con:
        con.execute("DELETE FROM events WHERE is_demo=?", (int(is_demo),))
        con.executemany(
            "INSERT INTO events(date,title,category,description,source,verified,is_demo) VALUES (?,?,?,?,?,?,?)",
            [(r["date"], r["title"], r.get("category", ""), r.get("description", ""), r.get("source", ""),
              int(r.get("verified", 0)), int(is_demo)) for r in rows],
        )
    return len(rows)


def load_events(is_demo: bool | None = None) -> list[dict]:
    q = "SELECT date,title,category,description,source,verified,is_demo FROM events"
    params: tuple = ()
    if is_demo is not None:
        q += " WHERE is_demo=?"
        params = (int(is_demo),)
    q += " ORDER BY date"
    with connect() as con:
        return [dict(r) for r in con.execute(q, params).fetchall()]


def clear_demo() -> None:
    with connect() as con:
        ids = [r["series_id"] for r in con.execute("SELECT series_id FROM series_meta WHERE is_demo=1")]
        for sid in ids:
            con.execute("DELETE FROM observations WHERE series_id=?", (sid,))
        con.execute("DELETE FROM series_meta WHERE is_demo=1")
        con.execute("DELETE FROM events WHERE is_demo=1")


def log_ingest(series_id: str, rows: int, status: str, message: str = "") -> None:
    with connect() as con:
        con.execute("INSERT INTO ingest_log(series_id,run_at,rows,status,message) VALUES (?,?,?,?,?)",
                    (series_id, _now(), rows, status, message[:500]))


def recent_ingest_log(limit: int = 30) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM ingest_log ORDER BY id DESC LIMIT ?", (limit,))]


def latest_problems(limit: int = 80) -> list[dict]:
    """Most recent log entry per series/source, kept only if it is an error or a warning."""
    seen, out = set(), []
    for r in recent_ingest_log(limit):          # newest first
        if r["series_id"] in seen:
            continue
        seen.add(r["series_id"])
        if r["status"] != "ok":
            out.append(r)
    return out


def data_mode() -> str:
    """'demo' | 'mixed' | 'live' | 'empty' — drives the warning banner in the UI."""
    meta = series_meta()
    if not meta:
        return "empty"
    flags = {bool(m["is_demo"]) for m in meta.values()}
    if flags == {True}:
        return "demo"
    if flags == {False}:
        return "live"
    return "mixed"


def db_stamp() -> float:
    """Change marker used to invalidate caches. With WAL, writes land in the -wal file first."""
    stamp = 0.0
    for p in (DB_PATH, DB_PATH.with_name(DB_PATH.name + "-wal")):
        try:
            stamp = max(stamp, p.stat().st_mtime)
        except FileNotFoundError:
            pass
    return stamp


def count_rows(series_id: str) -> int:
    """Stored rows of a REAL series (synthetic demo rows never count as history worth protecting)."""
    with connect() as con:
        return int(con.execute(
            "SELECT COUNT(*) FROM observations o JOIN series_meta m ON m.series_id=o.series_id "
            "WHERE o.series_id=? AND m.is_demo=0", (series_id,)).fetchone()[0])

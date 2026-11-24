"""Background refresh: pull live data at startup and then every N hours (needed on hosts like Render, where
the filesystem is wiped on each restart so the database must be rebuilt from the public sources)."""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Callable

from . import ingest

RETRY_AFTER_FAILURE_S = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Refresher:
    def __init__(self, interval_hours: float | None = None, after: Callable[[], None] | None = None):
        hours = interval_hours if interval_hours is not None else float(os.environ.get("ZTI_REFRESH_HOURS", "12"))
        self.interval_s = max(hours, 0.05) * 3600
        self.after = after            # e.g. warm the backtest cache once fresh data is in
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state: dict = dict(enabled=False, running=False, last_start=None, last_end=None,
                                last_ok=[], last_failed=[], last_error=None, interval_hours=hours)

    def run_once(self) -> dict | None:
        if not self._lock.acquire(blocking=False):   # a run is already in progress
            return None
        try:
            self.state.update(running=True, last_start=_now(), last_error=None)
            summary = ingest.run_live(quiet=True)
            self.state.update(last_ok=summary["ok"], last_failed=summary["failed"])
            if not summary["ok"]:
                self.state["last_error"] = "no series could be loaded (network blocked or sources down)"
            elif self.after:
                try:
                    self.after()
                except Exception as e:  # noqa: BLE001 - cache warming is optional
                    self.state["last_error"] = f"post-refresh step failed: {e!r}"
            return summary
        except Exception as e:  # noqa: BLE001
            self.state["last_error"] = repr(e)
            return None
        finally:
            self.state.update(running=False, last_end=_now())
            self._lock.release()

    def _loop(self) -> None:
        while not self._stop.is_set():
            summary = self.run_once()
            ok = bool(summary and summary["ok"])
            self._stop.wait(self.interval_s if ok else RETRY_AFTER_FAILURE_S)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.state["enabled"] = True
        self._thread = threading.Thread(target=self._loop, name="zti-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

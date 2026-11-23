"""Build the aligned modelling panel from stored series.

Rules that keep the maths honest:
  * Series published less often than the model frequency are *not* forward-filled into it (that would
    create fake zero-return weeks and a bogus jump). They are skipped and reported instead.
  * Event/step series (policy rate) are carried forward — a policy rate really does persist between meetings.
  * The final, incomplete period is dropped from the modelling panel (the latest data still shows in `asof`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import db
from ..config import FREQ_RANK, GROUPS, NATIVE_RANK, SERIES, TARGET, DEFAULT_FREQ


class DataError(RuntimeError):
    pass


@dataclass
class Panel:
    freq: str
    levels: pd.DataFrame            # period-end levels, columns = group names (TARGET + drivers)
    returns: pd.DataFrame           # log-returns (price) or first differences in pp (rate)
    used: dict[str, str]            # group -> series_id actually used
    kinds: dict[str, str]           # group -> 'price' | 'rate'
    skipped: list[dict] = field(default_factory=list)
    is_demo: bool = False
    asof: pd.Timestamp | None = None

    @property
    def drivers(self) -> list[str]:
        return [c for c in self.returns.columns if c != TARGET]


def _resample(s: pd.Series, native: str, freq: str) -> pd.Series:
    r = s.resample(freq).last()
    if native == "E":
        r = r.ffill()
    return r


def build_panel(freq: str = DEFAULT_FREQ) -> Panel:
    if freq not in FREQ_RANK:
        raise DataError(f"Unsupported frequency {freq!r}; use one of {sorted(FREQ_RANK)}")
    meta = db.series_meta()
    if TARGET not in meta:
        raise DataError("No USD/ZMW data loaded. Run `python -m app.ingest --demo` or `--live`.")
    need = FREQ_RANK[freq]
    cols: dict[str, pd.Series] = {}
    used: dict[str, str] = {}
    kinds: dict[str, str] = {}
    skipped: list[dict] = []
    for group, candidates in GROUPS.items():
        picked = None
        for sid in candidates:
            if sid not in meta:
                continue
            native = SERIES[sid]["native"]
            if NATIVE_RANK[native] > need:
                skipped.append(dict(series=sid, reason=f"published {native}-frequency; coarser than model frequency {freq}"))
                continue
            picked = sid
            break
        if picked is None:
            continue
        s = db.load_series(picked)
        if s.empty:
            continue
        cols[group] = _resample(s, SERIES[picked]["native"], freq)
        used[group] = picked
        kinds[group] = SERIES[picked]["kind"]

    levels = pd.DataFrame(cols).sort_index()
    levels = levels[levels[TARGET].notna() | (levels.index <= levels[TARGET].last_valid_index())]
    levels = levels.loc[levels[TARGET].first_valid_index():]

    # drop trailing incomplete period (its label is later than the last real observation)
    last_obs = db.load_series(used[TARGET]).index.max()
    if len(levels) and levels.index[-1] > last_obs:
        levels = levels.iloc[:-1]

    rets = pd.DataFrame(index=levels.index)
    for c in levels.columns:
        x = levels[c]
        rets[c] = np.log(x).diff() if kinds[c] == "price" else x.diff()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.iloc[1:]

    is_demo = bool(meta[used[TARGET]]["is_demo"])
    return Panel(freq=freq, levels=levels, returns=rets, used=used, kinds=kinds, skipped=skipped,
                 is_demo=is_demo, asof=last_obs)

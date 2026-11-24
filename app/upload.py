"""Turn a user-supplied CSV or Excel file into a (date -> value) series.

Built for real-world files such as a Bank of Zambia rate download: title rows above the header, dd/mm/yyyy dates,
thousands separators, several numeric columns (Buy / Sell / Mid). It picks the most plausible header row and
columns and tells the user which ones it used, so a wrong guess is visible; `value_col` overrides the guess.
"""
from __future__ import annotations

import csv
import io

import pandas as pd

MAX_BYTES = 5_000_000
PREFERRED = ("mid", "average", "avg", "rate", "close", "value", "price")
# non-ISO date layouts, day-first first (Zambian convention); the layout that parses the most rows wins
DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%d/%m/%y", "%d-%b-%y",
                "%m/%d/%Y", "%Y/%m/%d", "%b %d, %Y")
MAX_HEADER_ROW = 15
BLANKS = {"", "nan", "None", "NaT"}


class UploadError(ValueError):
    pass


def _read_raw(content: bytes, filename: str) -> pd.DataFrame:
    if len(content) > MAX_BYTES:
        raise UploadError("file is larger than 5 MB")
    if not content.strip():
        raise UploadError("file is empty")
    try:
        if filename.lower().endswith((".xlsx", ".xlsm")):
            return pd.read_excel(io.BytesIO(content), header=None, engine="openpyxl")
        text = content.decode("utf-8-sig", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
        # title rows above the header have fewer fields than the data, so parse by hand and pad
        delim = max((";", "\t", "|", ","), key=lambda d: sum(ln.count(d) for ln in lines[:200]))
        rows = list(csv.reader(lines, delimiter=delim))
        width = max(len(r) for r in rows)
        return pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str)
    except Exception as e:  # noqa: BLE001
        raise UploadError(f"could not read the file as CSV or Excel (.xlsx): {e}") from e


def _to_dates(col: pd.Series) -> pd.Series:
    """Best-effort date parse of a whole column (no threshold; the caller judges the parse rate)."""
    s = col.astype(str).str.strip()
    looks = s.str.contains(r"[-/ .:]|[A-Za-z]", regex=True)      # a bare number like 1850 is not a date
    best = pd.to_datetime(s, errors="coerce", format="ISO8601").where(looks)
    if best.notna().mean() >= 0.5:
        return best
    best_rate = best.notna().mean()
    for fmt in DATE_FORMATS:                                       # earlier (day-first) formats win ties
        parsed = pd.to_datetime(s, errors="coerce", format=fmt).where(looks)
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_rate, best = rate, parsed
    return best


def _numeric(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False).str.strip(),
                         errors="coerce")


def _blank(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip().isin(BLANKS)


def _rate(parsed: pd.Series, blank: pd.Series) -> tuple[float, int]:
    keep = ~blank
    n = int(keep.sum())
    return (float(parsed[keep].notna().mean()) if n else 0.0), n


def parse_upload(content: bytes, filename: str = "", value_col: str | None = None,
                 prefer: tuple[str, ...] = ()) -> tuple[pd.Series, dict]:
    raw = _read_raw(content, filename)
    if raw.shape[1] < 2:
        raise UploadError("need at least two columns: a date column and a value column")

    date_cols = [_to_dates(raw.iloc[:, i]) for i in range(raw.shape[1])]
    blanks = [_blank(raw.iloc[:, i]) for i in range(raw.shape[1])]
    best = None                                    # (rate, -header_row, -column), header_row, column
    for skip in range(0, min(MAX_HEADER_ROW, len(raw) - 1)):
        for i in range(raw.shape[1]):
            rate, n = _rate(date_cols[i].iloc[skip + 1:], blanks[i].iloc[skip + 1:])
            if n >= 3 and rate >= 0.8 and (best is None or (rate, -skip, -i) > best[0]):
                best = ((rate, -skip, -i), skip, i)
    if best is None:
        raise UploadError("could not find a date column (expected dates like 2026-11-20 or 20/11/2026)")
    _, skip, di = best

    names: list[str] = []
    for i, c in enumerate(raw.iloc[skip]):
        n = str(c).strip()
        n = f"column{i + 1}" if n in BLANKS else n
        names.append(f"{n}_{i + 1}" if n in names else n)
    body = raw.iloc[skip + 1:].reset_index(drop=True)
    body.columns = names
    dates = date_cols[di].iloc[skip + 1:].reset_index(drop=True)
    date_col = names[di]

    nums = {}
    for i, c in enumerate(names):
        if i == di:
            continue
        v = _numeric(body[c])
        rate, n = _rate(v, _blank(body[c]))
        if n >= 3 and rate >= 0.8:
            nums[c] = v
    if not nums:
        raise UploadError("could not find a numeric value column")
    if value_col:
        match = [c for c in nums if c.lower() == value_col.strip().lower()]
        if not match:
            raise UploadError(f"column {value_col!r} not found among numeric columns: {', '.join(nums)}")
        chosen = match[0]
    else:
        chosen = next((c for p in (*prefer, *PREFERRED) for c in nums if p in c.lower()), next(iter(nums)))

    s = pd.Series(nums[chosen].values, index=pd.DatetimeIndex(dates)).dropna()
    s = s[s.index.notna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        raise UploadError("no rows with both a valid date and a number")
    info = dict(date_column=date_col, value_column=chosen, numeric_columns=list(nums), rows=int(len(s)),
                first=str(s.index.min().date()), last=str(s.index.max().date()),
                sample=[dict(date=str(d.date()), value=float(v)) for d, v in s.tail(3).items()])
    return s, info

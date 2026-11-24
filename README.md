# Zambia Treasury Insight

Decision-support web app for a bank treasury: current market data, history explained, and forecasts that come with their own track record. **It does not trade.**

## Run it

```bash
pip install -r requirements.txt
python -m app.ingest --check     # dry run: fetch + validate every source, print a report, write nothing
python -m app.ingest --live      # real data (needs internet)
python -m app.ingest --demo      # OR synthetic data, clearly bannered, so everything works offline
uvicorn app.main:app --reload    # open http://localhost:8000
pytest                           # 65 tests
```

Hosting on Render: see [docs/DEPLOY_RENDER.md](docs/DEPLOY_RENDER.md). Set `ZTI_AUTO_INGEST=1` to fetch live data at
start-up and every `ZTI_REFRESH_HOURS`; set `ZTI_PASSWORD` to require a login.

## What's inside

| Tab | What it does |
|---|---|
| Overview | Latest values and changes; USD/ZMW chart with event markers; compare with any driver (rebased, one axis) |
| What happened & why | Pick any window → driver attribution that adds up exactly, unexplained remainder, biggest weeks, matching events, plain-language narrative |
| Forecast | Fan chart (50% / 90% ranges), model choice, driver "lean", full backtest table (error, skill vs random walk, t-stat, interval calibration) |
| Scenarios | "Copper −20%", "BoZ −100bp", custom shocks → typical kwacha response with 80/95% ranges; flags when recent and full-history sensitivities disagree |
| Data & quality | Provenance, freshness, what was excluded and why, rolling sensitivities, refresh status, ingest log |

## Data sources: what is real, what is verified

| Series | Source | How it gets in | Status |
|---|---|---|---|
| USD/ZMW | Bank of Zambia published average rate, via the open [Frankfurter API](https://frankfurter.dev/providers/boz/) | live | History starts Jan 2013 (kwacha rebased). Format confirmed from the provider's OpenAPI spec and sample data; **not yet run end-to-end** |
| US dollar index, Brent, US 10y, VIX, monthly copper | FRED public CSV | live | Standard endpoint; **not yet run end-to-end** |
| Daily copper | Yahoo HG=F → Stooq HG.F → your own `data/manual/COPPER_USD.csv` | live, best effort | Unofficial sources; swap in an LME feed for production |
| **BoZ policy rate, Apr 2012 – today** | Bank of Zambia Monetary Policy Report (Nov 2024), table "Monetary and Foreign Exchange Measures since April 2012"; 2025–26 from Reuters / Central Banking / Trading Economics | `data/manual/POLICY_RATE.csv` | Every "from" level chains to the previous row. 2025–26 dates ±1 day |
| **Annual inflation, Feb 2020 – Oct 2026** | ZamStats *The Monthly* (Mar 2025), Table 1.3; later months from press reports of ZamStats releases | `data/manual/CPI_YOY.csv` | Gaps: Apr, Jul, Aug 2025; nothing before Feb 2020 yet |
| Events | `data/events.csv` (curated) + every policy-rate change (derived automatically) | | `verified=0` rows need checking before the bank relies on them |

**Not covered yet** (no free machine-readable source found): T-bill and bond auction yields, FX reserves, Eurobond spread,
mining export receipts, Kariba/power data, pre-2020 inflation. Bank of Zambia's website is a JavaScript application
that cannot be scraped reliably, and its auction results are PDFs. Any of these can be added by dropping a
`data/manual/<SERIES_ID>.csv` (`date,value`) and registering the series in `app/config.py`; a Bloomberg/Refinitiv or
internal desk feed is one more connector returning a pandas Series.

## Getting data in when a feed fails

* **Data & quality → Load data** shows the exact error per source (URL, HTTP status), a **Retry** button, and a
  **file uploader** (CSV or Excel, e.g. a Bank of Zambia rates download). Columns are detected automatically and
  reported back; the dollar column is preferred for USD/ZMW; rows are merged into stored data.
* USD/ZMW is requested from the provider in one-year pieces, then as weekly averages, then from the provider's blended
  feed. The source label on the Data tab says which one was used.
* Permanent data: commit `data/manual/<SERIES_ID>.csv` (`date,value`); it is loaded on every start. For a series that has a
  live feed (e.g. USDZMW) it is used only when the feed fails, so a stale file never hides fresh data.
* On a public host (Render) the upload and retry controls stay locked until `ZTI_PASSWORD` is set.

## Safeguards in the data layer

* Every series passes `app/validate.py` before storage: unit/range checks, future dates dropped, 1000:1 kwacha rebasing
  repaired if old rows appear, large jumps flagged, and **a refresh can never shrink stored history**.
* Each source runs in isolation and is retried with backoff; failures are logged, shown in the app and never wipe data.
* Demo rows are only removed once real data has been stored, and the UI shows a banner whenever demo data is present.

## Design rules (why it behaves the way it does)

- **Explain first, predict second.** Attribution uses contemporaneous drivers; forecasts use only information known at the forecast date.
- **No fake precision.** Intervals are empirical quantiles; backtests check they contain the outcome ~90% of the time.
- **A model must earn its place.** `auto` picks the model with the best backtest error and falls back to the random walk if none beat it.
- **Low-frequency data is not forward-filled into weekly models.** Monthly inflation is used only in the monthly model; step series (policy rate) are carried forward; drivers with too little history are excluded and reported.
- **Associations, not causes.** Narratives say "associated with" and "coincided with".

## Next steps

1. Run `python -m app.ingest --check` and fix whatever fails.
2. Backfill inflation before 2020 and the missing 2025 months from ZamStats bulletins; add T-bill yields and reserves.
3. Tune and test models on the real history; keep only what beats the random walk in backtests.
4. Regime handling (pre/post 2020 default, 2022 IMF programme); more drivers.
5. Persistent database, audit log, model-governance pack.

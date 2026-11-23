# Zambia Treasury Insight

Decision-support web app for a bank treasury: current market data, history explained, and forecasts that come with their own track record. **It does not trade.**

## Run it

```bash
pip install -r requirements.txt
python -m app.ingest --demo      # synthetic data, so everything works immediately
python -m app.ingest --live      # real data (needs internet): BoZ FX, FRED, copper, data/manual/*.csv
uvicorn app.main:app --reload    # open http://localhost:8000
pytest                           # 23 tests
```

Running `--live` removes demo rows first, so synthetic and real data are never mixed. A banner stays on screen whenever demo data is loaded.

## What's inside

| Tab | What it does |
|---|---|
| Overview | Latest values and changes; USD/ZMW chart with event markers; compare with any driver (rebased, one axis) |
| What happened & why | Pick any window → driver attribution (copper, dollar, oil, rates, ...) that adds up exactly, unexplained remainder, biggest weeks, matching events, plain-language narrative |
| Forecast | Fan chart (50% / 90% ranges), model choice, driver "lean", full backtest table (error, skill vs random walk, t-stat, interval calibration) |
| Scenarios | "Copper −20%", "BoZ −100bp", custom shocks → typical kwacha response with 80/95% ranges; flags when recent and full-history sensitivities disagree |
| Data & quality | Provenance, freshness, what was excluded and why, rolling sensitivities, ingest log |

## Data sources

| Series | Source | Notes |
|---|---|---|
| USD/ZMW | Bank of Zambia published average rate via frankfurter.dev | Daily from 2006; derived by inverting BoZ's ZMW-base publication |
| Dollar index, Brent, US 10y, copper (monthly) | FRED public CSV | No API key |
| Copper (daily) | Yahoo HG=F, unofficial, best effort | Replace with an LME feed (Bloomberg/Refinitiv) in production |
| Policy rate, CPI, reserves, T-bill yields... | `data/manual/<SERIES_ID>.csv` (`date,value`) | **Seed files only cover 2025-26. Extend back to 2012+ from BoZ MPC statements and ZamStats.** |
| Events | `data/events.csv` | Curated; rows with `verified=0` need checking before the bank relies on them |

To add a driver: add it to `app/config.py` (`SERIES` and `GROUPS`), then either a connector or a manual CSV. Bloomberg/Refinitiv or internal desk data = one new connector returning a pandas Series.

## Design rules (why it behaves the way it does)

- **Explain first, predict second.** Attribution uses contemporaneous drivers; forecasts use only information known at the forecast date.
- **No fake precision.** Intervals are empirical quantiles; backtests check they contain the outcome ~90% of the time.
- **A model must earn its place.** `auto` picks the model with the best backtest error and falls back to the random walk if none beat it. On synthetic data it correctly falls back.
- **Low-frequency data is not forward-filled into weekly models.** Monthly CPI/reserves are used only in a monthly model; drivers with too little history are excluded and reported.
- **Associations, not causes.** Narratives say "associated with" and "coincided with".

## Known limits / next steps

1. Load real history for policy rate, CPI, reserves, T-bill/bond yields (biggest accuracy gain).
2. Add Eurobond spread, mining export receipts, power-supply (Kariba) data, fiscal calendar as drivers.
3. Regime switching (pre/post debt restructuring) and lead-lag tests; today's driver model is contemporaneous for explanation and 4-week-lagged for forecasting.
4. Move SQLite → PostgreSQL/TimescaleDB (only `app/db.py` changes); add authentication, audit log, scheduled ingestion.
5. Model governance pack: documentation, validation report, change log, user-visible model version.
6. Optional: LLM layer that *reads* computed results to write summaries (it must not invent causes).

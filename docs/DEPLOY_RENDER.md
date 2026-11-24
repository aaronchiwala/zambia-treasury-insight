# Deploying to Render

The repo contains a `render.yaml` Blueprint, so Render can configure everything itself.

## 0. Check the data feeds first (2 minutes, on your own computer)

```bash
git clone https://github.com/aaronchiwala/zambia-treasury-insight.git
cd zambia-treasury-insight
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.ingest --check        # fetches every source, validates, prints a report, writes nothing
```

Every line is `✓` (loaded), `!` (loaded, with a note to read) or `✗` (failed, with the reason). If a source fails,
send me the output and I will fix the connector before you deploy.

## 1. Create the service

1. Sign in at [render.com](https://render.com) and connect your GitHub account (grant it access to `zambia-treasury-insight`).
2. **New → Blueprint**, pick the `zambia-treasury-insight` repository, branch `main`.
3. Render reads `render.yaml` and shows one web service. It will ask for **`ZTI_PASSWORD`**: choose a strong password.
   The username is `treasury` (change `ZTI_USER` in the dashboard if you like). The password protects the whole site.
4. Click **Apply**. The first build takes a few minutes.

(Manual alternative: **New → Web Service**, runtime Python, build `pip install -r requirements.txt`,
start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check path `/api/health`,
and the environment variables listed in `render.yaml`.)

## 2. First start

Open the service URL and sign in. A "Loading live data" banner appears while the server fetches Bank of Zambia, FRED
and copper data (about a minute); the page reloads itself. Then open **Data & quality**: every series should show its
source, row count and date range. A failed source is listed there with the reason, and the server retries every 10 minutes.

## 3. Day-to-day

* **Updating the code:** push to `main` on GitHub; Render redeploys automatically.
* **Data refresh:** automatic at start and every `ZTI_REFRESH_HOURS` (12) hours.
* **After each Bank of Zambia MPC meeting that changes the rate:** add one line to `data/manual/POLICY_RATE.csv`
  (`2026-12-02,10.50`), commit, push. Monthly inflation: add a line to `data/manual/CPI_YOY.csv` when ZamStats publishes.

## 4. Things to know about the free plan

* Free instances sleep after a period of inactivity, and the first request afterwards is slow.
* The free filesystem is **not persistent**: the database is rebuilt from the public sources on every restart. That is
  why the app fetches data at start-up. If you later want instant start-up and stored history, move to a paid instance
  with a persistent disk (set `ZTI_DB=/var/data/zti.sqlite`) or to Postgres (only `app/db.py` changes).
* The first Forecast click can be slow on a small free CPU; the backtest is precomputed after each data refresh.
* Check Render's current plan limits before relying on it: they change.

## 5. Security

* Keep `ZTI_PASSWORD` set: without it the site is public. Render provides HTTPS.
* This tool uses only public data. Do not upload internal bank data to a public host; for that, deploy inside the bank's
  own infrastructure.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Build fails on Python version | In the Render dashboard set `PYTHON_VERSION` to a version Render lists as supported (3.11 to 3.13) |
| Banner says "Data not loaded" | Open **Data & quality** for per-source errors; USD/ZMW is the only essential series |
| Copper missing | Yahoo/Stooq are unofficial; add `data/manual/COPPER_USD.csv` (`date,value`, USD/tonne) from an LME export |
| 401 on every request | Wrong `ZTI_USER`/`ZTI_PASSWORD`; `/api/health` is always open for Render's checks |

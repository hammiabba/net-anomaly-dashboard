# Net Anomaly Dashboard

Detect anomalies in network telemetry — and **prove** the detector works.

This project generates realistic, *labeled* synthetic network telemetry
(latency, packet loss, throughput, error rate), compares an Isolation Forest
detector against a statistical Z-score baseline with real precision/recall/F1
numbers, serves detections over a FastAPI API, and visualizes everything in a
Streamlit dashboard with a live-stream replay mode.

## Problem

Network operations teams drown in metrics. A link degrades slowly over two
hours, or latency spikes for 60 seconds at 3 AM — per-metric static thresholds
either miss the slow failures or page engineers for noise. This project asks:
can a multivariate model catch what univariate thresholds miss, and can we
*measure* the difference instead of hand-waving about it?

## Architecture

```
┌──────────────┐     CSV      ┌──────────────┐    train/predict   ┌───────────┐
│  telemetry/  │ ──────────► │   detector/  │ ◄───────────────── │    api/   │
│  generator   │  labeled    │ IF + Z-score │                    │  FastAPI  │
│  (seeded)    │  1-min      │ evaluate.py  │                    │ /detect   │
└──────────────┘  samples    └──────┬───────┘                    └───────────┘
                                    │ scores
                                    ▼
                             ┌──────────────┐
                             │  dashboard/  │
                             │  Streamlit + │
                             │  Plotly      │
                             └──────────────┘
```

Data flows one way: the generator produces labeled telemetry → detectors train
on the past and score the present → the API serves scores → the dashboard
visualizes them. The API and dashboard share the same detector code, so what
you see in the UI is exactly what the API returns.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate 72h of labeled telemetry (deterministic: same seed, same data)
python -m telemetry.generate --hours 72 --anomalies 12 --out data/telemetry.csv

# 2. Train + compare detectors (prints the precision/recall/F1 report)
python -m detector.evaluate --hours 168 --anomalies 30 --seed 7

# 3. Serve detections
uvicorn api.app:app --port 8000
# POST /detect with {"points": [...], "detector": "isolation_forest"}

# 4. Open the dashboard
streamlit run dashboard/app.py
```

## How detection works

Both detectors train **unsupervised** on the first 70% of the series
(chronological split — in production you only ever train on the past) and are
evaluated on the last 30% against the injected labels.

- **Isolation Forest** (primary): an ensemble that isolates observations by
  random partitioning. Points with short average isolation path lengths are
  anomalous. It models the *joint* distribution across all four metrics, so a
  subtle simultaneous drift in latency + loss + throughput looks anomalous even
  when no single metric crosses a threshold.
- **Z-score** (baseline): flags a point when any single metric deviates more
  than 3σ from its training mean. Simple, explainable — and blind to
  cross-metric patterns.

## Measured results

`python -m detector.evaluate --hours 168 --anomalies 30 --seed 7`
(70/30 chronological split, IF contamination=0.08, Z threshold=3.0):

| detector         | precision | recall | F1    |
|------------------|-----------|--------|-------|
| isolation_forest | 1.000     | 0.564  | 0.722 |
| zscore           | 1.000     | 0.388  | 0.559 |

Recall broken down by anomaly type — this is where the story lives:

| detector         | spike | shift | drift |
|------------------|-------|-------|-------|
| isolation_forest | 1.00  | 1.00  | 0.36  |
| zscore           | 1.00  | 0.99  | 0.11  |

Both detectors nail spikes and sustained shifts. The Z-score baseline goes
nearly blind on **gradual drift** (recall 0.11): a slow ramp never deviates far
enough on any single metric at any single moment. Isolation Forest catches 3×
more of it (0.36) because the *combination* of small deviations across metrics
is unusual in the joint distribution. Neither is perfect on drift — the early
ramp genuinely looks normal, which is also true in production.

## Key design decisions

1. **Isolation Forest over pure thresholds.** Not because it's fancier — because
   the measurement above says so. On identical data, IF achieves F1 0.722 vs
   0.559 for Z-score, with the entire gap coming from drift detection. The
   dashboard lets you flip between the two detectors and watch the Z-score miss
   drifts in real time.

2. **Labeled synthetic data instead of "looks right" evaluation.** Anomaly
   detection is usually evaluated by eyeballing charts, which proves nothing.
   Injecting labeled anomalies (spike / shift / drift) turns it into a
   supervised benchmark: precision, recall, F1, and per-type recall. The labels
   are the reason the comparison above is a measurement, not an opinion.

3. **Seeded, deterministic generation.** Same seed → byte-identical dataset, on
   any machine. That makes the evaluation reproducible, the tests meaningful
   (determinism is asserted in `tests/test_generator.py`), and the API's
   startup training reproducible.

4. **Chronological train/test split.** No shuffling. Shuffling time series
   leaks the future into training and inflates scores — the 70/30 split mirrors
   how the model would actually be deployed.

5. **Small API surface.** Two endpoints (`/health`, `/detect`), strict pydantic
   validation, batch input. A detection service should be boring and correct.

## Project structure

```
telemetry/      Synthetic generator: diurnal patterns + noise, labeled
                spike/shift/drift injection, CLI (python -m telemetry.generate)
detector/       IsolationForestDetector + ZScoreDetector (train/predict/score,
                joblib save/load), evaluate.py comparison report
api/            FastAPI service: POST /detect (batch, validated), GET /health
dashboard/      Streamlit app: Plotly charts, alert table, metric cards,
                live-stream replay mode
tests/          pytest: generator determinism, detector F1, API contract
.github/        CI: install + pytest on 3.11/3.12, push and PR
data/           Generated CSVs (gitignored)
models/         Saved detectors (gitignored)
```

## Demo it in 2 minutes

1. `streamlit run dashboard/app.py` → sidebar: 72 hours, 12 anomalies, seed 42.
2. Point at the red X markers on the latency chart — those are detections, and
   the F1 card shows the score against ground-truth labels.
3. Flip the detector to `zscore` and watch drift anomalies disappear from the
   chart while spikes stay — the core finding, live.
4. Tick **"Simulate live stream"** → Replay: the last 4 hours replay as a live
   feed with alerts firing.
5. `curl -X POST localhost:8000/detect` with a spiky point (see API section
   above) → `"is_anomaly": true`.

## Tech

Python 3.11+ · scikit-learn (Isolation Forest) · FastAPI + pydantic · Streamlit
+ Plotly · pandas/numpy · pytest · GitHub Actions. No API keys, no external
services — everything runs locally.

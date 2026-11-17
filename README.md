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

# 2b. Compare every setup (raw vs daily pattern removed vs robust Z), 20 seeds
python -m detector.evaluate --compare --seeds 20

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

## Update: a fairer baseline

After I posted the results above, a friend pointed out that the Z-score was set
up to lose. Two things were wrong with it:

1. **The daily pattern.** Throughput swings by about 440 Mbps over a day. Z-score
   treats that whole range as normal, so a slow drift hides inside it. Fix:
   learn the typical value for each hour of the day from the training window
   and subtract it (`detector/preprocess.py`).
2. **Anomalies in the training data.** The injected spikes, shifts and drifts
   in the first 70% blow up the standard deviation, so the 3σ line ends up far
   too high. Fix: a robust Z-score that uses the median and MAD (median
   absolute deviation) instead of the mean and std.

`python -m detector.evaluate --compare --seeds 20` (averaged over 20 simulated
weeks, same settings as above):

| setup                 | detector         | F1    | drift recall | false alarms |
|-----------------------|------------------|-------|--------------|--------------|
| raw data              | isolation_forest | 0.721 | 0.41         | 0.0          |
| raw data              | zscore           | 0.490 | 0.17         | 0.0          |
| daily pattern removed | isolation_forest | 0.754 | 0.46         | 0.6          |
| daily pattern removed | zscore           | 0.602 | 0.24         | 0.0          |
| robust Z-score        | zscore           | 0.818 | 0.58         | 0.1          |
| both fixes            | isolation_forest | 0.754 | 0.46         | 0.6          |
| both fixes            | zscore           | 0.913 | 0.79         | 5.5          |

So on raw data with default settings, Isolation Forest beats the textbook
Z-score. With both fixes, the simple Z-score beats Isolation Forest clearly,
and it kept winning when I also tuned Isolation Forest's contamination on a
validation window. How the data was prepared mattered more than which method
was used.

Part of this is the synthetic data itself: every injected drift pushes each
metric clearly in one direction, which is exactly what a per-metric check is
good at. Isolation Forest's strength (odd combinations of metrics that are each
normal on their own) never really gets tested here. Real traffic is the next
step.

Both fixes are toggles in the dashboard sidebar ("Remove daily pattern",
"Robust Z-score") and flags on the evaluation script (`--deseason`,
`--robust-z`). Defaults are unchanged, so the original numbers above still
reproduce.

## Key design decisions

1. **Measure, then pick the detector.** The first version picked Isolation
   Forest because it beat a raw Z-score (F1 0.722 vs 0.559). The fairer
   comparison in the update above flipped that. Keeping both detectors behind
   the same interface is what made re-running the comparison cheap.

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

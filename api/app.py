"""FastAPI anomaly detection service.

Trains both detectors once at startup on deterministic synthetic data
(same seed every boot, so scores are reproducible) and serves them over:

- ``GET  /health`` — liveness probe.
- ``POST /detect`` — score a batch of telemetry points.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI

from detector.detectors import IsolationForestDetector, ZScoreDetector
from telemetry.generator import FEATURE_COLUMNS, generate_telemetry

from .schemas import DetectRequest, DetectResponse, DetectionResult

# Training config: 2 weeks of 1-minute samples, fixed seed -> reproducible model.
TRAIN_HOURS = 24 * 14
TRAIN_ANOMALIES = 40
TRAIN_SEED = 123


@lru_cache(maxsize=1)
def get_detectors() -> dict[str, IsolationForestDetector | ZScoreDetector]:
    """Train (once) and cache both detectors on synthetic baseline data."""
    df = generate_telemetry(hours=TRAIN_HOURS, n_anomalies=TRAIN_ANOMALIES, seed=TRAIN_SEED)
    X = df[FEATURE_COLUMNS]
    detectors: dict[str, IsolationForestDetector | ZScoreDetector] = {
        "isolation_forest": IsolationForestDetector(contamination=0.08).train(X),
        "zscore": ZScoreDetector(threshold=3.0).train(X),
    }
    return detectors


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_detectors()  # warm up so the first request isn't slow
    yield


app = FastAPI(
    title="Net Anomaly Detection API",
    description="Score network telemetry batches for anomalies.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "features": FEATURE_COLUMNS}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    detectors = get_detectors()
    detector = detectors[request.detector]
    X = pd.DataFrame(
        [
            {
                "latency_ms": p.latency_ms,
                "packet_loss_pct": p.packet_loss_pct,
                "throughput_mbps": p.throughput_mbps,
                "error_rate_pct": p.error_rate_pct,
            }
            for p in request.points
        ]
    )
    scores = detector.score(X)
    flags = detector.predict(X)
    results = [
        DetectionResult(timestamp=p.timestamp, score=float(s), is_anomaly=bool(f))
        for p, s, f in zip(request.points, scores, flags)
    ]
    return DetectResponse(detector=request.detector, results=results)

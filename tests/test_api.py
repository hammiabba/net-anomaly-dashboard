"""Tests for the FastAPI detection service."""

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def _point(**overrides):
    base = {
        "timestamp": datetime(2026, 1, 10, 12, 0, 0).isoformat(),
        "latency_ms": 45.0,
        "packet_loss_pct": 0.2,
        "throughput_mbps": 800.0,
        "error_rate_pct": 0.05,
    }
    base.update(overrides)
    return base


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_happy_path():
    points = [_point(timestamp=(datetime(2026, 1, 10, 12, 0, 0)
                                + timedelta(minutes=i)).isoformat())
              for i in range(10)]
    r = client.post("/detect", json={"points": points})
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "isolation_forest"
    assert len(body["results"]) == 10
    for res in body["results"]:
        assert isinstance(res["score"], float)
        assert isinstance(res["is_anomaly"], bool)


def test_detect_zscore_detector_choice():
    r = client.post("/detect", json={"points": [_point()], "detector": "zscore"})
    assert r.status_code == 200
    assert r.json()["detector"] == "zscore"


def test_detect_extreme_point_flagged():
    # A blatant spike should be flagged by at least the z-score detector.
    r = client.post("/detect", json={
        "points": [_point(latency_ms=900.0, packet_loss_pct=8.0,
                          throughput_mbps=100.0, error_rate_pct=5.0)],
        "detector": "zscore",
    })
    assert r.status_code == 200
    assert r.json()["results"][0]["is_anomaly"] is True


def test_detect_validation_errors():
    # Negative latency violates the schema.
    r = client.post("/detect", json={"points": [_point(latency_ms=-5)]})
    assert r.status_code == 422
    # Empty batch violates min_length.
    r = client.post("/detect", json={"points": []})
    assert r.status_code == 422
    # Unknown detector violates the Literal.
    r = client.post("/detect", json={"points": [_point()], "detector": "nope"})
    assert r.status_code == 422

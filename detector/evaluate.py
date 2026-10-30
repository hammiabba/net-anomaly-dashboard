"""Evaluate both detectors on labeled synthetic telemetry.

Trains on the first 70% of the series (chronological split — no shuffling, as
in production you only ever train on the past) and evaluates on the last 30%.
Prints precision/recall/F1 overall plus recall broken down by anomaly type,
which is where the Isolation Forest vs. Z-score story shows up: thresholds
catch spikes but go blind on slow drifts.

Example:
    python -m detector.evaluate --hours 168 --anomalies 30 --seed 7
"""

from __future__ import annotations

import argparse

from sklearn.metrics import precision_recall_fscore_support

from telemetry.generator import FEATURE_COLUMNS, LABEL_COLUMN, generate_telemetry

from .detectors import IsolationForestDetector, ZScoreDetector


def evaluate(
    hours: float = 168,
    n_anomalies: int = 30,
    seed: int = 7,
    contamination: float = 0.08,
    z_threshold: float = 3.0,
    verbose: bool = True,
) -> dict:
    df = generate_telemetry(hours=hours, n_anomalies=n_anomalies, seed=seed)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[LABEL_COLUMN].to_numpy()

    detectors = {
        "isolation_forest": IsolationForestDetector(contamination=contamination),
        "zscore": ZScoreDetector(threshold=z_threshold),
    }
    results: dict = {}
    for name, det in detectors.items():
        det.train(X_train)
        pred = det.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        by_type = {}
        for kind in ("spike", "shift", "drift"):
            mask = (test["anomaly_type"] == kind).to_numpy()
            if mask.sum():
                by_type[kind] = float(pred[mask].mean())  # recall within this type
        results[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "recall_by_type": by_type,
            "n_test": len(test), "n_anomalous_test": int(y_test.sum()),
        }

    if verbose:
        print(f"train={len(train)} rows, test={len(test)} rows "
              f"({int(y_test.sum())} anomalous in test)\n")
        header = f"{'detector':<18}{'precision':>10}{'recall':>10}{'f1':>10}"
        print(header)
        print("-" * len(header))
        for name, r in results.items():
            print(f"{name:<18}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}")
        print("\nrecall by anomaly type:")
        for name, r in results.items():
            parts = "  ".join(f"{k}={v:.2f}" for k, v in r["recall_by_type"].items())
            print(f"  {name:<16} {parts}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detectors.")
    parser.add_argument("--hours", type=float, default=168)
    parser.add_argument("--anomalies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    args = parser.parse_args()
    evaluate(
        hours=args.hours, n_anomalies=args.anomalies, seed=args.seed,
        contamination=args.contamination, z_threshold=args.z_threshold,
    )


if __name__ == "__main__":
    main()

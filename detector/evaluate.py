"""Evaluate both detectors on labeled synthetic telemetry.

Trains on the first 70% of the series (chronological split, no shuffling, since
in production you only ever train on the past) and evaluates on the last 30%.
Prints precision/recall/F1 overall plus recall broken down by anomaly type.

Two optional fixes make the Z-score baseline fair:

- ``--deseason``: subtract the daily pattern (learned from the training window)
  before detection, for both detectors.
- ``--robust-z``: Z-score uses median + MAD instead of mean + std, so anomalies
  inside the training window don't inflate the spread.

``--seeds N`` averages over N different simulated weeks, so one lucky seed
can't decide the result. ``--compare`` runs every setup and prints one table.

Examples:
    python -m detector.evaluate --hours 168 --anomalies 30 --seed 7
    python -m detector.evaluate --deseason --robust-z --seeds 20
    python -m detector.evaluate --compare --seeds 20
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from telemetry.generator import FEATURE_COLUMNS, LABEL_COLUMN, generate_telemetry

from .detectors import IsolationForestDetector, ZScoreDetector
from .preprocess import DailyProfile

ANOMALY_KINDS = ("spike", "shift", "drift")


def _run_once(
    hours: float,
    n_anomalies: int,
    seed: int,
    contamination: float,
    z_threshold: float,
    deseason: bool,
    robust_z: bool,
) -> dict:
    df = generate_telemetry(hours=hours, n_anomalies=n_anomalies, seed=seed)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]

    if deseason:
        profile = DailyProfile(FEATURE_COLUMNS).fit(train)
        X_train, X_test = profile.transform(train), profile.transform(test)
    else:
        X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_test = test[LABEL_COLUMN].to_numpy()

    detectors = {
        "isolation_forest": IsolationForestDetector(contamination=contamination),
        "zscore": ZScoreDetector(threshold=z_threshold, robust=robust_z),
    }
    results: dict = {}
    for name, det in detectors.items():
        det.train(X_train)
        pred = det.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        by_type = {}
        for kind in ANOMALY_KINDS:
            mask = (test["anomaly_type"] == kind).to_numpy()
            if mask.sum():
                by_type[kind] = float(pred[mask].mean())  # recall within this type
        results[name] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "false_alarms": int(((pred == 1) & (y_test == 0)).sum()),
            "recall_by_type": by_type,
            "n_test": len(test), "n_anomalous_test": int(y_test.sum()),
        }
    return results


def _average(runs: list[dict]) -> dict:
    out: dict = {}
    for name in runs[0]:
        rs = [r[name] for r in runs]
        out[name] = {
            k: float(np.mean([r[k] for r in rs]))
            for k in ("precision", "recall", "f1", "false_alarms")
        }
        out[name]["recall_by_type"] = {
            kind: float(np.mean([r["recall_by_type"][kind] for r in rs
                                 if kind in r["recall_by_type"]]))
            for kind in ANOMALY_KINDS
            if any(kind in r["recall_by_type"] for r in rs)
        }
        out[name]["n_test"] = rs[0]["n_test"]
        out[name]["n_anomalous_test"] = int(np.mean([r["n_anomalous_test"] for r in rs]))
    return out


def evaluate(
    hours: float = 168,
    n_anomalies: int = 30,
    seed: int = 7,
    contamination: float = 0.08,
    z_threshold: float = 3.0,
    deseason: bool = False,
    robust_z: bool = False,
    n_seeds: int = 1,
    verbose: bool = True,
) -> dict:
    """Run the comparison. With ``n_seeds > 1``, uses seeds ``seed..seed+n-1``
    and returns averages."""
    runs = [
        _run_once(hours, n_anomalies, s, contamination, z_threshold, deseason, robust_z)
        for s in range(seed, seed + n_seeds)
    ]
    results = runs[0] if n_seeds == 1 else _average(runs)

    if verbose:
        setup = []
        if deseason:
            setup.append("daily pattern removed")
        if robust_z:
            setup.append("robust Z-score")
        any_r = next(iter(results.values()))
        print(f"setup: {', '.join(setup) or 'raw data, default settings'}")
        print(f"seeds: {seed}..{seed + n_seeds - 1}"
              + (" (averaged)" if n_seeds > 1 else ""))
        print(f"test window: {any_r['n_test']} rows, "
              f"~{any_r['n_anomalous_test']} anomalous\n")
        header = (f"{'detector':<18}{'precision':>10}{'recall':>10}"
                  f"{'f1':>10}{'false alarms':>14}")
        print(header)
        print("-" * len(header))
        for name, r in results.items():
            print(f"{name:<18}{r['precision']:>10.3f}{r['recall']:>10.3f}"
                  f"{r['f1']:>10.3f}{r['false_alarms']:>14.1f}")
        print("\nrecall by anomaly type:")
        for name, r in results.items():
            parts = "  ".join(f"{k}={v:.2f}" for k, v in r["recall_by_type"].items())
            print(f"  {name:<16} {parts}")
    return results


SETUPS = (
    ("raw data", False, False),
    ("daily pattern removed", True, False),
    ("robust Z-score", False, True),
    ("both fixes", True, True),
)


def compare(
    hours: float = 168,
    n_anomalies: int = 30,
    seed: int = 0,
    n_seeds: int = 20,
    contamination: float = 0.08,
    z_threshold: float = 3.0,
    verbose: bool = True,
) -> dict:
    """Run every setup over the same seeds and print one table."""
    table = {
        label: evaluate(hours, n_anomalies, seed, contamination, z_threshold,
                        deseason=ds, robust_z=rz, n_seeds=n_seeds, verbose=False)
        for label, ds, rz in SETUPS
    }
    if verbose:
        print(f"averaged over seeds {seed}..{seed + n_seeds - 1}\n")
        header = (f"{'setup':<24}{'detector':<18}{'f1':>7}"
                  f"{'drift recall':>14}{'false alarms':>14}")
        print(header)
        print("-" * len(header))
        for label, res in table.items():
            for name, r in res.items():
                drift = r["recall_by_type"].get("drift", float("nan"))
                print(f"{label:<24}{name:<18}{r['f1']:>7.3f}"
                      f"{drift:>14.2f}{r['false_alarms']:>14.1f}")
            print()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detectors.")
    parser.add_argument("--hours", type=float, default=168)
    parser.add_argument("--anomalies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None,
                        help="first seed (default 7, or 0 with --compare)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="number of seeds to average (default 1, or 20 with --compare)")
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--deseason", action="store_true",
                        help="remove the daily pattern before detection")
    parser.add_argument("--robust-z", action="store_true",
                        help="Z-score with median + MAD instead of mean + std")
    parser.add_argument("--compare", action="store_true",
                        help="run every setup and print one comparison table")
    args = parser.parse_args()

    if args.compare:
        compare(
            hours=args.hours, n_anomalies=args.anomalies,
            seed=0 if args.seed is None else args.seed,
            n_seeds=20 if args.seeds is None else args.seeds,
            contamination=args.contamination, z_threshold=args.z_threshold,
        )
    else:
        evaluate(
            hours=args.hours, n_anomalies=args.anomalies,
            seed=7 if args.seed is None else args.seed,
            contamination=args.contamination, z_threshold=args.z_threshold,
            deseason=args.deseason, robust_z=args.robust_z,
            n_seeds=1 if args.seeds is None else args.seeds,
        )


if __name__ == "__main__":
    main()

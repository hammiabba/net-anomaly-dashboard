"""CLI: generate a labeled telemetry CSV.

Example:
    python -m telemetry.generate --hours 72 --anomalies 12 --out data/telemetry.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled network telemetry.")
    parser.add_argument("--hours", type=float, default=72, help="Hours of 1-minute samples.")
    parser.add_argument("--anomalies", type=int, default=12, help="Number of injected anomalies.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (deterministic output).")
    parser.add_argument("--out", type=str, default="data/telemetry.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = generate_telemetry(hours=args.hours, n_anomalies=args.anomalies, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_flagged = int(df["is_anomaly"].sum())
    print(f"Wrote {len(df)} rows to {out} "
          f"({args.anomalies} injected anomalies, {n_flagged} labeled points).")


if __name__ == "__main__":
    main()

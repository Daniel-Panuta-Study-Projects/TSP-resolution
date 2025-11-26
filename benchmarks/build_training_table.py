from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import extract_features
from tsp.data import load_cities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine benchmark results into a feature+winner training table."
    )
    parser.add_argument(
        "--instances-dir",
        default="data/instances",
        help="Directory containing the generated CSV instances.",
    )
    parser.add_argument(
        "--raw-results",
        default="benchmarks/raw_results.csv",
        help="CSV with per-strategy distances/times.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/training_dataset.csv",
        help="Path to the feature table with winner labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_training_dataset(
        instances_dir=Path(args.instances_dir),
        raw_results_csv=Path(args.raw_results),
        output_csv=Path(args.output),
    )


def build_training_dataset(
    instances_dir: Path,
    raw_results_csv: Path,
    output_csv: Path,
) -> None:
    """Read raw benchmarks, identify winners, and append feature vectors."""
    results = _load_raw_results(raw_results_csv)
    if not results:
        raise FileNotFoundError(
            f"Raw results CSV {raw_results_csv} is empty or missing."
        )

    winners = _pick_winners(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = None
        for instance_name, winning_strategy in winners.items():
            csv_path = instances_dir / instance_name
            cities = load_cities(csv_path)
            features = extract_features(cities)

            row = {"instance_name": instance_name, "winner": winning_strategy}
            row.update(features)

            if writer is None:
                fieldnames = ["instance_name", "winner"] + sorted(features.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(row)
    print(f"Training dataset saved to {output_csv}")


def _load_raw_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _pick_winners(rows: list[dict]) -> dict[str, str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["instance_name"]].append(row)

    winners: dict[str, str] = {}
    for instance_name, items in grouped.items():
        best = min(items, key=lambda r: float(r["distance"]))
        winners[instance_name] = best["strategy"]
    return winners


if __name__ == "__main__":
    main()

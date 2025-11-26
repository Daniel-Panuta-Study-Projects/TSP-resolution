from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsp.data import load_cities
from tsp.solver import TSPSolver

from ml.strategies import STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each heuristic strategy on every generated TSP instance."
    )
    parser.add_argument(
        "--instances-dir",
        default="data/instances",
        help="Directory containing CSV instances (e.g., cities_001.csv).",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/raw_results.csv",
        help="CSV file where raw per-strategy results will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed applied per strategy run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmarks(
        instances_dir=Path(args.instances_dir),
        output_csv=Path(args.output),
        base_seed=args.seed,
    )


def run_benchmarks(
    instances_dir: Path,
    output_csv: Path,
    base_seed: int | None,
) -> None:
    """Execute every strategy on each CSV instance and record metrics."""
    instance_files = sorted(instances_dir.glob("*.csv"))
    if not instance_files:
        raise FileNotFoundError(
            f"No CSV instances found in {instances_dir}. Generate them first."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["instance_name", "strategy", "distance", "time_seconds"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for inst_idx, csv_path in enumerate(instance_files):
            cities = load_cities(csv_path)
            solver = TSPSolver(cities)
            print(f"Benchmarking {csv_path.name} with {len(cities)} cities...")

            for strat_idx, (label, params) in enumerate(STRATEGIES.items()):
                kwargs = dict(params)
                if base_seed is not None:
                    kwargs["seed"] = base_seed + inst_idx * len(STRATEGIES) + strat_idx

                start = time.perf_counter()
                solution = solver.solve(**kwargs)
                elapsed = time.perf_counter() - start

                writer.writerow(
                    {
                        "instance_name": csv_path.name,
                        "strategy": label,
                        "distance": f"{solution.distance:.6f}",
                        "time_seconds": f"{elapsed:.6f}",
                    }
                )
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_strategies import run_benchmarks
from .build_training_table import build_training_dataset
from .generate_instances import generate_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark pipeline: generate instances, benchmark strategies, build training dataset."
    )
    parser.add_argument("--output-dir", default="data/instances", help="Where generated CSV instances are stored.")
    parser.add_argument("--samples", type=int, default=20, help="How many instances to generate.")
    parser.add_argument("--min-cities", type=int, default=15, help="Minimum number of cities per instance.")
    parser.add_argument("--max-cities", type=int, default=60, help="Maximum number of cities per instance.")
    parser.add_argument("--width", type=float, default=120.0, help="Maximum X coordinate value.")
    parser.add_argument("--height", type=float, default=120.0, help="Maximum Y coordinate value.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base random seed for reproducibility.")
    parser.add_argument("--raw-results", default="benchmarks/raw_results.csv", help="CSV path for per-strategy metrics.")
    parser.add_argument("--training-dataset", default="benchmarks/training_dataset.csv", help="CSV path for features + winner labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1: Generating random instances...")
    generate_instances(
        output_dir=Path(args.output_dir),
        samples=args.samples,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    print("\nStep 2: Benchmarking strategies...")
    run_benchmarks(
        instances_dir=Path(args.output_dir),
        output_csv=Path(args.raw_results),
        base_seed=args.seed,
    )

    print("\nStep 3: Building training dataset...")
    build_training_dataset(
        instances_dir=Path(args.output_dir),
        raw_results_csv=Path(args.raw_results),
        output_csv=Path(args.training_dataset),
    )

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()

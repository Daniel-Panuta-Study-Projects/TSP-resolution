from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random TSP instances saved as CSV files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/instances",
        help="Directory where cities_*.csv files will be written.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="How many CSV instances to generate.",
    )
    parser.add_argument(
        "--min-cities",
        type=int,
        default=15,
        help="Minimum number of cities per instance.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=60,
        help="Maximum number of cities per instance.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=120.0,
        help="Maximum X coordinate for generated cities.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=120.0,
        help="Maximum Y coordinate for generated cities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_instances(
        output_dir=Path(args.output_dir),
        samples=args.samples,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )


def generate_instances(
    output_dir: Path,
    samples: int,
    min_cities: int,
    max_cities: int,
    width: float,
    height: float,
    seed: int | None,
) -> None:
    """Create multiple CSV files with random city coordinates."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(samples):
        city_count = rng.randint(min_cities, max_cities)
        rows = [
            {
                "name": f"N{idx}_{city_index}",
                "x": f"{rng.uniform(0, width):.4f}",
                "y": f"{rng.uniform(0, height):.4f}",
            }
            for city_index in range(city_count)
        ]

        file_path = output_dir / f"cities_{idx + 1:03d}.csv"
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "x", "y"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Generated {file_path} ({city_count} cities)")


if __name__ == "__main__":
    main()

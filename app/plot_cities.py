from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsp.data import load_cities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display cities from a CSV in a simple 2D scatter plot."
    )
    parser.add_argument(
        "--data",
        default="data/cities_50.csv",
        help="CSV file containing columns name,x,y.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Annotate each point with the city name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.data)
    cities = load_cities(path)

    xs = [city.x for city in cities]
    ys = [city.y for city in cities]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, color="tab:blue", s=40)

    if args.show_labels:
        for city in cities:
            ax.annotate(
                city.name,
                (city.x, city.y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_title(f"Cities from {path.name}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

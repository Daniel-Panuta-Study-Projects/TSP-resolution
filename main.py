from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tsp.data import load_cities
from tsp.solver import TSPSolver

DEFAULT_DATASET = Path("data/cities_50.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heuristic solvers for a Travelling Salesman Problem instance.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to CSV containing columns name,x,y (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--method",
        choices=["nearest_neighbor", "greedy", "random"],
        default="nearest_neighbor",
        help="Constructive heuristic used to build the initial tour.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the starting city (used by nearest_neighbor).",
    )
    parser.add_argument(
        "--apply-2opt",
        dest="apply_two_opt_flag",
        action="store_true",
        help="Apply 2-opt local search to the current tour.",
    )
    parser.add_argument(
        "--annealing",
        action="store_true",
        help="Run Simulated Annealing starting from the current tour.",
    )
    parser.add_argument(
        "--ga",
        action="store_true",
        help="Run the Genetic Algorithm seeded with the current tour.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible stochastic heuristics.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, num_cities: int) -> None:
    if args.start < 0 or args.start >= num_cities:
        raise ValueError(
            f"Start index {args.start} is outside the valid range [0, {num_cities - 1}]",
        )


def main() -> int:
    args = parse_args()
    data_path = args.data

    if not data_path.exists():
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        return 1

    cities = load_cities(data_path)
    validate_args(args, len(cities))

    solver = TSPSolver(cities)
    solution = solver.solve(
        method=args.method,
        start=args.start,
        apply_two_opt_flag=args.apply_two_opt_flag,
        annealing=args.annealing,
        genetic=args.ga,
        seed=args.seed,
    )

    route_names = solution.as_city_names(cities)

    print("=== TSP Heuristic Toolkit ===")
    print(f"Cities: {len(cities)}")
    print(f"Dataset: {data_path}")
    print(f"Method: {args.method}")
    print(f"2-opt: {'yes' if args.apply_two_opt_flag else 'no'}")
    print(f"Annealing: {'yes' if args.annealing else 'no'}")
    print(f"Genetic Algorithm: {'yes' if args.ga else 'no'}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Total distance: {solution.distance:.3f}")
    print("Tour (by index):", " -> ".join(map(str, solution.route_indices)))
    print("Tour (by name):", " -> ".join(route_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

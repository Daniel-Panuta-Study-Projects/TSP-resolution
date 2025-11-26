from __future__ import annotations  # Allow type hints to reference later declarations.

import random  # Random is used for reproducibility controls and stochastic solvers.
from dataclasses import dataclass  # dataclass simplifies Solution container.
from typing import List, Sequence  # Typing helpers clarify API contracts.

from .data import City, build_distance_matrix  # Import domain objects and utilities.
from .heuristics import (
    genetic_algorithm,
    greedy,
    nearest_neighbor,
    random_tour,
    simulated_annealing,
    tour_length,
    two_opt,
)  # Bring in actual heuristic implementations.


@dataclass
class Solution:
    """Bundle storing a route and its total distance."""

    route_indices: List[int]
    distance: float

    def as_city_names(self, cities: Sequence[City]) -> List[str]:
        """Return the tour as readable city names."""
        return [cities[idx].name for idx in self.route_indices]


class TSPSolver:
    """High-level helper that wires together the heuristics."""

    def __init__(self, cities: Sequence[City]):
        self.cities = list(cities)  # Copy user-provided cities for internal use.
        self.distance_matrix = build_distance_matrix(self.cities)  # Precompute speeds up heuristics.

    def solve(
        self,
        method: str = "nearest_neighbor",
        start: int = 0,
        apply_two_opt_flag: bool = False,
        annealing: bool = False,
        genetic: bool = False,
        seed: int | None = None,
    ) -> Solution:
        """Execute selected heuristics and return the resulting route."""
        if seed is not None:
            random.seed(seed)  # Ensure runs are reproducible when requested.

        base_route = self._construct_route(method, start)  # Build initial tour.

        if apply_two_opt_flag:
            base_route = two_opt(base_route, self.distance_matrix)  # Deterministic refinement.

        if annealing:
            base_route = simulated_annealing(base_route, self.distance_matrix)  # Stochastic improvement.

        if genetic:
            base_route = genetic_algorithm(
                self.distance_matrix,
                base_tour=base_route,
            )
            if apply_two_opt_flag:
                base_route = two_opt(base_route, self.distance_matrix, max_iterations=500)  # Re-run local search.

        length = tour_length(base_route, self.distance_matrix)  # Evaluate final distance.
        return Solution(route_indices=base_route, distance=length)

    def _construct_route(self, method: str, start: int) -> List[int]:
        """Dispatch to the requested constructive heuristic."""
        method = method.lower()
        if method == "nearest_neighbor":
            return nearest_neighbor(self.distance_matrix, start=start)
        if method == "greedy":
            return greedy(self.distance_matrix)
        if method == "random":
            return random_tour(len(self.cities))
        raise ValueError(f"Unknown method: {method}")

"""Convenience exports for the TSP heuristic toolkit."""

from .data import City, build_distance_matrix, load_cities  # Expose core data helpers.
from .solver import TSPSolver  # Export the high-level solver interface.

__all__ = ["City", "build_distance_matrix", "load_cities", "TSPSolver"]  # Public API.

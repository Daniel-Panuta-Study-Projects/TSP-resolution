from __future__ import annotations  # Postpone evaluation of annotations for typing.

import csv  # CSV helps parse the coordinate dataset shared via .csv files.
from dataclasses import dataclass  # dataclass keeps City representation concise.
from pathlib import Path  # Path provides handy filesystem helpers.
from typing import Iterable, List  # Typing info clarifies expected collections.


@dataclass(frozen=True)
class City:
    """Represents a named point in 2D space used by TSP solvers."""

    name: str
    x: float
    y: float


def load_cities(csv_path: str | Path) -> List[City]:
    """Read a CSV file and return a list of City objects populated with data."""
    path = Path(csv_path)  # Ensure we are working with a proper Path object.
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)  # DictReader maps each row using header names.
        cities = [
            City(row["name"], float(row["x"]), float(row["y"]))
            for row in reader  # Convert each CSV row into a City instance.
        ]
    if not cities:
        raise ValueError(f"No cities found in {path}")  # Guard against empty files.
    return cities  # Return the fully populated list for downstream solvers.


def build_distance_matrix(cities: Iterable[City]) -> List[List[float]]:
    """Build a full matrix of pairwise Euclidean distances between cities."""
    city_list = list(cities)  # Materialize iterable because we need random access.
    n = len(city_list)  # Number of cities defines the matrix dimensions.
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]  # Initialize zeros.
    for i in range(n):
        matrix[i][i] = 0.0  # Distance from a city to itself is zero.
        for j in range(i + 1, n):
            dist = _euclidean(city_list[i], city_list[j])  # Compute unique pair.
            matrix[i][j] = dist  # Store computed distance (upper triangle).
            matrix[j][i] = dist  # Mirror value to lower triangle for symmetry.
    return matrix  # Precomputed matrix accelerates repeated distance lookups.


def _euclidean(a: City, b: City) -> float:
    """Return Euclidean distance between two cities."""
    dx = a.x - b.x  # Horizontal difference.
    dy = a.y - b.y  # Vertical difference.
    return (dx * dx + dy * dy) ** 0.5  # Pythagorean distance formula.

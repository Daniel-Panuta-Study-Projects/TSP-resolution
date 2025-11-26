from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

from tsp.data import City, build_distance_matrix


def extract_features(
    cities: Sequence[City],
    dist_matrix: Sequence[Sequence[float]] | None = None,
) -> Dict[str, float]:
    """Compute simple descriptors describing a TSP instance."""
    if dist_matrix is None:
        dist_matrix = build_distance_matrix(cities)

    n = len(cities)
    xs = [city.x for city in cities]
    ys = [city.y for city in cities]

    pairwise = _collect_pairwise(dist_matrix)
    if not pairwise:
        raise ValueError("Feature extraction requires at least two cities.")

    nearest = [
        min(dist_matrix[i][j] for j in range(n) if j != i)
        for i in range(n)
    ]

    features: Dict[str, float] = {
        "num_cities": float(n),
        "x_span": max(xs) - min(xs),
        "y_span": max(ys) - min(ys),
        "bbox_area": (max(xs) - min(xs)) * (max(ys) - min(ys)),
        "mean_distance": statistics.mean(pairwise),
        "median_distance": statistics.median(pairwise),
        "min_distance": min(pairwise),
        "max_distance": max(pairwise),
        "std_distance": statistics.pstdev(pairwise),
        "avg_nearest_neighbor": statistics.mean(nearest),
    }
    return features


def _collect_pairwise(
    dist_matrix: Sequence[Sequence[float]],
) -> List[float]:
    distances: List[float] = []
    n = len(dist_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(dist_matrix[i][j])
    return distances

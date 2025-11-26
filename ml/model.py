from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


FeatureVector = Dict[str, float]
LabeledRow = Tuple[FeatureVector, str]


@dataclass
class CentroidModel:
    """Lightweight classifier that assigns the closest normalized centroid."""

    feature_names: List[str]
    feature_mean: List[float]
    feature_std: List[float]
    centroids: Dict[str, List[float]]
    counts: Dict[str, int]

    def predict(self, features: FeatureVector) -> str:
        vector = self._normalize(features)
        best_label = None
        best_distance = float("inf")
        for label, centroid in self.centroids.items():
            distance = _euclidean(vector, centroid)
            if distance < best_distance:
                best_distance = distance
                best_label = label
        if best_label is None:
            raise ValueError("Model has no centroids trained.")
        return best_label

    def _normalize(self, features: FeatureVector) -> List[float]:
        values: List[float] = []
        for idx, name in enumerate(self.feature_names):
            if name not in features:
                raise KeyError(f"Missing feature '{name}' for prediction")
            raw = features[name]
            norm = (raw - self.feature_mean[idx]) / self.feature_std[idx]
            values.append(norm)
        return values


def train_centroid_model(rows: Iterable[LabeledRow]) -> CentroidModel:
    dataset = list(rows)
    if not dataset:
        raise ValueError("Dataset is empty; cannot train model.")

    feature_names = sorted(dataset[0][0].keys())
    matrix = [[row[0][name] for name in feature_names] for row in dataset]
    mean = _column_mean(matrix)
    std = _column_std(matrix, mean)

    sums: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}

    for features, label in dataset:
        normalized = [
            (features[name] - mean[idx]) / std[idx]
            for idx, name in enumerate(feature_names)
        ]
        counts[label] = counts.get(label, 0) + 1
        sums.setdefault(label, [0.0] * len(normalized))
        for idx, value in enumerate(normalized):
            sums[label][idx] += value

    centroids = {
        label: [value / counts[label] for value in vector]
        for label, vector in sums.items()
    }

    return CentroidModel(
        feature_names=feature_names,
        feature_mean=mean,
        feature_std=std,
        centroids=centroids,
        counts=counts,
    )


def save_model(model: CentroidModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as fh:
        pickle.dump(model, fh)


def load_model(path: str | Path) -> CentroidModel:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _column_mean(matrix: Sequence[Sequence[float]]) -> List[float]:
    cols = len(matrix[0])
    sums = [0.0] * cols
    for row in matrix:
        for idx, value in enumerate(row):
            sums[idx] += value
    return [value / len(matrix) for value in sums]


def _column_std(
    matrix: Sequence[Sequence[float]],
    mean: Sequence[float],
) -> List[float]:
    cols = len(matrix[0])
    sums = [0.0] * cols
    for row in matrix:
        for idx, value in enumerate(row):
            diff = value - mean[idx]
            sums[idx] += diff * diff
    return [max(math.sqrt(value / len(matrix)), 1e-9) for value in sums]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

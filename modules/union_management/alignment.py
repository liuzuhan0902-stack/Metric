"""Vector-space alignment primitives used by union management.

The implementation deliberately depends only on NumPy.  Orthogonal mappings are
used when dimensions match because they preserve Euclidean distances between
vectors.  A least-squares mapping is available for spaces with different
dimensions.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SpaceAlignment:
    """An affine row-vector mapping from one embedding space to another."""

    source_space: str
    target_space: str
    matrix: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    method: str = "orthogonal"

    @classmethod
    def fit(
        cls,
        source_vectors: np.ndarray,
        target_vectors: np.ndarray,
        source_space: str,
        target_space: str,
        method: str = "orthogonal",
    ) -> "SpaceAlignment":
        """Learn a mapping from paired anchor vectors.

        Rows at the same position in ``source_vectors`` and ``target_vectors``
        must describe the same entity.
        """
        source = _as_matrix(source_vectors, "source_vectors")
        target = _as_matrix(target_vectors, "target_vectors")
        if source.shape[0] != target.shape[0] or source.shape[0] == 0:
            raise ValueError("Alignment requires the same non-zero number of anchors")

        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        source_centered = source - source_mean
        target_centered = target - target_mean

        if method == "orthogonal":
            if source.shape[1] != target.shape[1]:
                raise ValueError("Orthogonal alignment requires equal dimensions")
            left, _, right_t = np.linalg.svd(
                source_centered.T @ target_centered, full_matrices=False
            )
            matrix = left @ right_t
        elif method == "least_squares":
            matrix = np.linalg.lstsq(
                source_centered, target_centered, rcond=None
            )[0]
        else:
            raise ValueError(f"Unsupported alignment method: {method}")

        return cls(
            str(source_space),
            str(target_space),
            np.asarray(matrix, dtype=float),
            np.asarray(source_mean, dtype=float),
            np.asarray(target_mean, dtype=float),
            method,
        )

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Transform one vector or a matrix of vectors into the target space."""
        array = np.asarray(vectors, dtype=float)
        was_vector = array.ndim == 1
        matrix = _as_matrix(array, "vectors")
        if matrix.shape[1] != self.matrix.shape[0]:
            raise ValueError(
                f"Expected vectors with dimension {self.matrix.shape[0]}, "
                f"received {matrix.shape[1]}"
            )
        transformed = (matrix - self.source_mean) @ self.matrix + self.target_mean
        return transformed[0] if was_vector else transformed

    def transform_query(self, query_vector: np.ndarray) -> np.ndarray:
        """Transform a query vector; equivalent to transforming a data vector."""
        return self.transform(query_vector)


class AlignmentManager:
    """Registry for directed mappings between multiple embedding spaces."""

    def __init__(self):
        self._alignments: Dict[Tuple[str, str], SpaceAlignment] = {}

    def register(self, alignment: SpaceAlignment) -> SpaceAlignment:
        self._alignments[(alignment.source_space, alignment.target_space)] = alignment
        return alignment

    def fit(
        self,
        source_vectors: np.ndarray,
        target_vectors: np.ndarray,
        source_space: str,
        target_space: str,
        method: str = "orthogonal",
    ) -> SpaceAlignment:
        return self.register(
            SpaceAlignment.fit(
                source_vectors,
                target_vectors,
                source_space,
                target_space,
                method,
            )
        )

    def get(self, source_space: str, target_space: str) -> Optional[SpaceAlignment]:
        return self._alignments.get((str(source_space), str(target_space)))

    def transform(
        self,
        vectors: np.ndarray,
        source_space: str,
        target_space: str,
    ) -> np.ndarray:
        """Transform vectors, using identity only when source and target match."""
        if str(source_space) == str(target_space):
            return np.asarray(vectors, dtype=float).copy()
        alignment = self.get(source_space, target_space)
        if alignment is None:
            raise ValueError(
                f"No alignment registered from '{source_space}' to '{target_space}'"
            )
        return alignment.transform(vectors)


def _as_matrix(vectors: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty vector or two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array

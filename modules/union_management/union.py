from core.base import MetricRelation, VectorSpace
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from modules.union_management.alignment import AlignmentManager, SpaceAlignment


class UnifiedMetricRelation(MetricRelation):
    """Materialised relation produced by a hard union.

    It implements the existing ``MetricRelation`` read interface and adds an
    exact ``search`` method so the offline result can be queried directly.
    """

    def __init__(
        self,
        name: str,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        vector_space: VectorSpace,
        alignment_manager: AlignmentManager,
    ):
        self.name = name
        self.vectors = np.asarray(vectors, dtype=float)
        self.metadata = metadata
        self.ids = [item["id"] for item in metadata]
        self.vector_space = vector_space
        self.alignment_manager = alignment_manager
        self._id_to_index = {item_id: index for index, item_id in enumerate(self.ids)}

    def get_vectors(self, ids: List[str]) -> np.ndarray:
        return self.vectors[[self._id_to_index[item_id] for item_id in ids]]

    def get_metadata(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [self.metadata[self._id_to_index[item_id]] for item_id in ids]

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        query_space: Optional[str] = None,
    ) -> List[str]:
        """Run the online phase against the one materialised vector matrix."""
        return [
            result["id"]
            for result in self.search_with_scores(query_vector, k, query_space)
        ]

    def search_with_scores(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        query_space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        target_space = self.vector_space.model_name
        query_space = query_space or target_space
        query = self.alignment_manager.transform(
            query_vector, query_space, target_space
        )
        return _exact_search(self, query, k)


class UnionManager:
    """
    Module 3: Virtual and Hard Unions (Assigned to: Student 3)
    
    Objective: Implement mechanisms to query across multiple vector relations
    that may exist in different embedding spaces.
    
    Task for Student 3:
    1. Soft Union: Query translation without re-indexing.
    2. Hard Union: Physical data integration into a common space.
    """
    def __init__(self, alignment_manager: Optional[AlignmentManager] = None):
        self.alignment_manager = alignment_manager or AlignmentManager()

    def register_alignment(self, alignment: SpaceAlignment) -> SpaceAlignment:
        """Register a learned mapping for subsequent soft and hard unions."""
        return self.alignment_manager.register(alignment)

    def fit_alignment(
        self,
        source_vectors: np.ndarray,
        target_vectors: np.ndarray,
        source_space: str,
        target_space: str,
        method: str = "orthogonal",
    ) -> SpaceAlignment:
        """Fit and register an alignment using paired anchor vectors."""
        return self.alignment_manager.fit(
            source_vectors, target_vectors, source_space, target_space, method
        )

    def soft_union_query(
        self,
        relations: List[MetricRelation],
        query_vector: np.ndarray,
        k: int = 10,
        query_space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a search across multiple relations without physically merging them.
        
        Task for Student 3:
        Implement logic to handle cases where 'relations' might use different models.
        How do you aggregate results? How do you translate the query?
        """
        if not relations or k <= 0:
            return []

        query = np.asarray(query_vector, dtype=float)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        query_space = query_space or _space_name(relations[0])

        candidates: List[Dict[str, Any]] = []
        for relation_index, relation in enumerate(relations):
            target_space = _space_name(relation)
            translated_query = self.alignment_manager.transform(
                query, query_space, target_space
            )
            relation_results = _exact_search(relation, translated_query, k)
            for rank, result in enumerate(relation_results, start=1):
                result["source_relation"] = _relation_name(relation, relation_index)
                result["space"] = target_space
                result["local_rank"] = rank
                candidates.append(result)

        # Orthogonal alignment preserves Euclidean scale, so merging by distance
        # produces the true global top-k rather than taking one result per database.
        # The remaining keys make ties deterministic and retain local-rank evidence.
        candidates.sort(
            key=lambda item: (
                item["distance"],
                item["local_rank"],
                item["source_relation"],
                item["id"],
            )
        )
        return candidates[:k]

    def hard_union_integrate(
        self,
        relations: List[MetricRelation],
        target_space: Optional[str] = None,
        name: str = "HardUnion",
    ) -> MetricRelation:
        """
        Physically merge multiple relations into a single new MetricRelation.
        
        Task for Student 3:
        Implement a strategy to re-embed or align data from different relations 
        into a unified vector space.
        """
        if not relations:
            raise ValueError("Hard union requires at least one relation")
        target_space = target_space or _space_name(relations[0])

        vector_batches: List[np.ndarray] = []
        combined_metadata: List[Dict[str, Any]] = []
        target_dimension: Optional[int] = None

        for relation_index, relation in enumerate(relations):
            ids = _relation_ids(relation)
            vectors = np.asarray(relation.get_vectors(ids), dtype=float)
            transformed = self.alignment_manager.transform(
                vectors, _space_name(relation), target_space
            )
            if target_dimension is None:
                target_dimension = transformed.shape[1]
            elif transformed.shape[1] != target_dimension:
                raise ValueError("Aligned relations do not share a target dimension")

            source_name = _relation_name(relation, relation_index)
            source_metadata = relation.get_metadata(ids)
            for source_id, metadata in zip(ids, source_metadata):
                item = dict(metadata)
                item.update(
                    {
                        "id": f"{source_name}::{source_id}",
                        "source_id": source_id,
                        "source_relation": source_name,
                        "source_space": _space_name(relation),
                    }
                )
                combined_metadata.append(item)
            vector_batches.append(transformed)

        vectors = np.vstack(vector_batches)
        space = VectorSpace(target_space, int(target_dimension or 0))
        return UnifiedMetricRelation(
            name, vectors, combined_metadata, space, self.alignment_manager
        )


def _relation_ids(relation: MetricRelation) -> List[str]:
    if hasattr(relation, "get_ids"):
        return list(relation.get_ids())
    if hasattr(relation, "ids"):
        return list(relation.ids)
    metadata = getattr(relation, "metadata", None)
    if isinstance(metadata, Sequence):
        return [item["id"] for item in metadata]
    raise TypeError(
        "MetricRelation must expose ids/get_ids to participate in a union"
    )


def _space_name(relation: MetricRelation) -> str:
    space = getattr(relation, "vector_space", None)
    if space is not None:
        return str(space.model_name)
    vectors = getattr(relation, "vectors", None)
    if vectors is None:
        ids = _relation_ids(relation)
        if not ids:
            raise ValueError("Cannot infer the space of an empty relation")
        vectors = relation.get_vectors(ids[:1])
    dimension = np.asarray(vectors).shape[-1]
    # Legacy relations do not describe their model. Treat equal-dimensional legacy
    # relations as one shared space to preserve the starter project's behaviour.
    return f"legacy-dimension-{dimension}"


def _relation_name(relation: MetricRelation, fallback_index: int = 0) -> str:
    return str(getattr(relation, "name", f"relation_{fallback_index}"))


def _exact_search(
    relation: MetricRelation, query_vector: np.ndarray, k: int
) -> List[Dict[str, Any]]:
    ids = _relation_ids(relation)
    if not ids or k <= 0:
        return []
    vectors = np.asarray(relation.get_vectors(ids), dtype=float)
    query = np.asarray(query_vector, dtype=float)
    if vectors.ndim != 2 or query.ndim != 1 or vectors.shape[1] != query.shape[0]:
        raise ValueError("Query and relation vector dimensions do not match")
    distances = np.linalg.norm(vectors - query, axis=1)
    order = sorted(range(len(ids)), key=lambda i: (float(distances[i]), ids[i]))
    return [
        {"id": ids[index], "distance": float(distances[index])}
        for index in order[: min(k, len(order))]
    ]

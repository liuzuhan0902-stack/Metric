"""Reproducible demonstration of virtual and hard vector-database unions."""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import VectorSpace
from main import SimpleMetricRelation
from modules.union_management.union import UnionManager


def build_relation(name, space_name, vectors):
    metadata = [{"id": f"item_{index}"} for index in range(len(vectors))]
    relation = SimpleMetricRelation(name, vectors, metadata)
    relation.vector_space = VectorSpace(space_name, vectors.shape[1])
    return relation


def main():
    rng = np.random.default_rng(7)
    source_vectors = rng.normal(size=(20, 4))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    target_vectors = source_vectors @ orthogonal + np.array([1.0, -2.0, 0.5, 3.0])

    source = build_relation("database_a", "space_a", source_vectors)
    target = build_relation("database_b", "space_b", target_vectors)
    manager = UnionManager()
    manager.fit_alignment(source_vectors, target_vectors, "space_a", "space_b")
    manager.fit_alignment(target_vectors, source_vectors, "space_b", "space_a")

    query = source_vectors[5]
    virtual_results = manager.soft_union_query(
        [source, target], query, k=4, query_space="space_a"
    )
    hard_union = manager.hard_union_integrate(
        [source, target], target_space="space_a"
    )
    hard_results = hard_union.search(query, k=4, query_space="space_a")

    print("Virtual union results:")
    for result in virtual_results:
        print(result)
    print("\nHard union results:")
    for result in hard_results:
        print(result)


if __name__ == "__main__":
    main()

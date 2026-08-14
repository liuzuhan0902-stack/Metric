"""Reproducible benchmark for single, virtual, and hard vector unions.

The synthetic databases share a latent semantic space.  Each observed embedding
space is produced by an independent orthogonal transform and translation, so
paired anchors have known correspondence while database records remain disjoint.
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import VectorSpace
from main import SimpleMetricRelation
from modules.union_management.alignment import SpaceAlignment
from modules.union_management.union import UnionManager


@dataclass
class BenchmarkDataset:
    relations: List[SimpleMetricRelation]
    latent_vectors: List[np.ndarray]
    queries: np.ndarray
    query_vectors: np.ndarray
    anchor_views: List[np.ndarray]


def generate_multispace_data(
    database_count: int,
    vectors_per_database: int,
    dimension: int,
    query_count: int,
    seed: int,
) -> BenchmarkDataset:
    """Generate related databases as views of one latent semantic space."""
    if database_count < 1 or vectors_per_database < 1 or dimension < 2:
        raise ValueError("database_count, vector count, and dimension must be positive")
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(max(2 * dimension, 32), dimension))
    queries = rng.normal(size=(query_count, dimension))
    relations: List[SimpleMetricRelation] = []
    latent_batches: List[np.ndarray] = []
    anchor_views: List[np.ndarray] = []

    for database_index in range(database_count):
        latent = rng.normal(size=(vectors_per_database, dimension))
        raw_matrix = rng.normal(size=(dimension, dimension))
        orthogonal, _ = np.linalg.qr(raw_matrix)
        translation = rng.normal(scale=2.0, size=dimension)
        observed = latent @ orthogonal + translation
        observed_anchors = anchors @ orthogonal + translation
        metadata = [
            {"id": f"item_{item_index}", "latent_database": database_index}
            for item_index in range(vectors_per_database)
        ]
        relation = SimpleMetricRelation(
            f"database_{database_index}", observed, metadata
        )
        relation.vector_space = VectorSpace(f"space_{database_index}", dimension)
        relations.append(relation)
        latent_batches.append(latent)
        anchor_views.append(observed_anchors)

    # Queries are supplied in Space A, matching the public union-query contract.
    first_anchors = anchor_views[0]
    alignment = SpaceAlignment.fit(anchors, first_anchors, "latent", "space_0")
    query_vectors = alignment.transform(queries)
    return BenchmarkDataset(
        relations, latent_batches, queries, query_vectors, anchor_views
    )


def configured_manager(
    dataset: BenchmarkDataset, aligned: bool
) -> UnionManager:
    """Create either learned alignments or an explicit no-alignment baseline."""
    manager = UnionManager()
    dimension = dataset.query_vectors.shape[1]
    source_anchors = dataset.anchor_views[0]
    for index, target_anchors in enumerate(dataset.anchor_views):
        target_space = f"space_{index}"
        if aligned:
            manager.fit_alignment(
                source_anchors, target_anchors, "space_0", target_space
            )
            manager.fit_alignment(
                target_anchors, source_anchors, target_space, "space_0"
            )
        elif index != 0:
            identity = np.eye(dimension)
            zeros = np.zeros(dimension)
            manager.register_alignment(
                SpaceAlignment(
                    "space_0", target_space, identity, zeros, zeros, "identity"
                )
            )
            manager.register_alignment(
                SpaceAlignment(
                    target_space, "space_0", identity, zeros, zeros, "identity"
                )
            )
    return manager


def ground_truth(dataset: BenchmarkDataset, k: int) -> List[set]:
    all_vectors = np.vstack(dataset.latent_vectors)
    keys = [
        (f"database_{database_index}", f"item_{item_index}")
        for database_index, vectors in enumerate(dataset.latent_vectors)
        for item_index in range(len(vectors))
    ]
    truth = []
    for query in dataset.queries:
        distances = np.linalg.norm(all_vectors - query, axis=1)
        order = np.argsort(distances, kind="stable")[:k]
        truth.append({keys[index] for index in order})
    return truth


def recall_at_k(results: Sequence[set], truth: Sequence[set], k: int) -> float:
    return float(np.mean([len(found & expected) / k for found, expected in zip(results, truth)]))


def alignment_errors(dataset: BenchmarkDataset) -> Tuple[float, float]:
    """Return anchor RMSE in Space A before and after learned alignment."""
    reference = dataset.anchor_views[0]
    before, after = [], []
    manager = configured_manager(dataset, aligned=True)
    for index, anchors in enumerate(dataset.anchor_views[1:], start=1):
        before.append(np.sqrt(np.mean((anchors - reference) ** 2)))
        restored = manager.alignment_manager.transform(
            anchors, f"space_{index}", "space_0"
        )
        after.append(np.sqrt(np.mean((restored - reference) ** 2)))
    return float(np.mean(before)), float(np.mean(after))


def timed_queries(callable_search, query_vectors: np.ndarray) -> Tuple[List, float]:
    results = []
    start = time.perf_counter()
    for query in query_vectors:
        results.append(callable_search(query))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return results, elapsed_ms / len(query_vectors)


def run_configuration(
    database_count: int,
    vectors_per_database: int,
    dimension: int,
    query_count: int,
    k: int,
    seed: int,
) -> List[Dict[str, object]]:
    dataset = generate_multispace_data(
        database_count, vectors_per_database, dimension, query_count, seed
    )
    truth = ground_truth(dataset, k)
    before_error, after_error = alignment_errors(dataset)
    common = {
        "database_count": database_count,
        "vectors_per_database": vectors_per_database,
        "total_vectors": database_count * vectors_per_database,
        "dimension": dimension,
        "query_count": query_count,
        "k": k,
        "seed": seed,
    }
    rows: List[Dict[str, object]] = []

    # Single-database exact search is the coverage baseline: it cannot retrieve
    # relevant vectors stored only in other databases.
    relation = dataset.relations[0]
    single_results, latency = timed_queries(
        lambda query: _single_search(relation, query, k), dataset.query_vectors
    )
    rows.append(
        _result_row(common, "single", "not_applicable", latency, 0.0,
                    recall_at_k(single_results, truth, k), "")
    )

    for aligned in (False, True):
        state = "after" if aligned else "before"
        error = after_error if aligned else before_error
        manager = configured_manager(dataset, aligned)
        soft_results, latency = timed_queries(
            lambda query: _soft_keys(
                manager.soft_union_query(
                    dataset.relations, query, k=k, query_space="space_0"
                )
            ),
            dataset.query_vectors,
        )
        rows.append(
            _result_row(common, "soft", state, latency, 0.0,
                        recall_at_k(soft_results, truth, k), error)
        )

        build_start = time.perf_counter()
        unified = manager.hard_union_integrate(
            dataset.relations, target_space="space_0"
        )
        build_ms = (time.perf_counter() - build_start) * 1000.0
        hard_results, latency = timed_queries(
            lambda query: _hard_keys(
                unified.search(query, k=k, query_space="space_0"), unified
            ),
            dataset.query_vectors,
        )
        rows.append(
            _result_row(common, "hard", state, latency, build_ms,
                        recall_at_k(hard_results, truth, k), error)
        )
    return rows


def _single_search(relation, query: np.ndarray, k: int) -> set:
    distances = np.linalg.norm(relation.vectors - query, axis=1)
    indices = np.argsort(distances, kind="stable")[:k]
    return {(relation.name, relation.ids[index]) for index in indices}


def _soft_keys(results: Iterable[Dict[str, object]]) -> set:
    return {(result["source_relation"], result["id"]) for result in results}


def _hard_keys(ids: Iterable[str], relation) -> set:
    metadata = relation.get_metadata(list(ids))
    return {(item["source_relation"], item["source_id"]) for item in metadata}


def _result_row(common, method, state, latency, build, recall, error):
    return {
        **common,
        "method": method,
        "alignment_state": state,
        "query_latency_ms": round(latency, 6),
        "build_latency_ms": round(build, 6),
        "recall_at_k": round(recall, 6),
        "alignment_rmse": "" if error == "" else round(float(error), 8),
    }


def write_results(rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--databases", default="2,3,5")
    parser.add_argument("--vectors", default="100,500")
    parser.add_argument("--dimensions", default="16,32")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", default="experiments/results/union_benchmark.csv"
    )
    args = parser.parse_args()

    rows = []
    for database_count in parse_int_list(args.databases):
        for vector_count in parse_int_list(args.vectors):
            for dimension in parse_int_list(args.dimensions):
                rows.extend(
                    run_configuration(
                        database_count,
                        vector_count,
                        dimension,
                        args.queries,
                        args.k,
                        args.seed,
                    )
                )
    write_results(rows, args.output)
    print(f"Wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()

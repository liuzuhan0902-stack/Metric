import unittest
import numpy as np
import os
import json
from main import SimpleMetricRelation
from core.base import MetricRelation, VectorSpace
from modules.union_management.alignment import SpaceAlignment
from modules.union_management.union import UnifiedMetricRelation, UnionManager

class TestUnionManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use generated data for testing
        data_dir = "data"
        vector_path = os.path.join(data_dir, "vectors.npy")
        metadata_path = os.path.join(data_dir, "metadata.json")
        
        if os.path.exists(vector_path) and os.path.exists(metadata_path):
            cls.vectors = np.load(vector_path)
            with open(metadata_path, "r") as f:
                cls.metadata = json.load(f)
            cls.relation = SimpleMetricRelation("TestRelation", cls.vectors, cls.metadata)
            cls.dim = cls.vectors.shape[1]
        else:
            cls.dim = 128
            cls.vectors = np.random.rand(10, cls.dim).astype(np.float32)
            cls.metadata = [{"id": f"item_{i}"} for i in range(10)]
            cls.relation = SimpleMetricRelation("MockRelation", cls.vectors, cls.metadata)

    def setUp(self):
        self.manager = UnionManager()

    def test_soft_union_interface(self):
        """Test if soft_union_query accepts a list of relations."""
        query_vector = np.random.rand(self.dim).astype(np.float32)
        try:
            results = self.manager.soft_union_query([self.relation], query_vector)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"soft_union_query failed with {e}")

    def test_hard_union_interface(self):
        """Test if hard_union_integrate accepts a list of relations."""
        try:
            result = self.manager.hard_union_integrate([self.relation])
            self.assertIsInstance(result, MetricRelation)
            self.assertEqual(len(result.ids), len(self.relation.ids))
        except Exception as e:
            self.fail(f"hard_union_integrate failed with {e}")


class TestVectorSpaceUnions(unittest.TestCase):
    def setUp(self):
        self.source_vectors = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=float
        )
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.target_vectors = self.source_vectors @ rotation + np.array([4.0, -3.0])
        metadata = [{"id": f"entity_{i}"} for i in range(4)]
        self.source = SimpleMetricRelation("source_db", self.source_vectors, metadata)
        self.source.vector_space = VectorSpace("source_space", 2)
        self.target = SimpleMetricRelation("target_db", self.target_vectors, metadata)
        self.target.vector_space = VectorSpace("target_space", 2)

        self.manager = UnionManager()
        self.manager.fit_alignment(
            self.source_vectors,
            self.target_vectors,
            "source_space",
            "target_space",
        )
        self.manager.fit_alignment(
            self.target_vectors,
            self.source_vectors,
            "target_space",
            "source_space",
        )

    def test_orthogonal_alignment_preserves_pairwise_distances(self):
        alignment = self.manager.alignment_manager.get(
            "source_space", "target_space"
        )
        transformed = alignment.transform(self.source_vectors)
        np.testing.assert_allclose(transformed, self.target_vectors, atol=1e-10)
        original_distances = np.linalg.norm(
            self.source_vectors[:, None] - self.source_vectors[None, :], axis=2
        )
        transformed_distances = np.linalg.norm(
            transformed[:, None] - transformed[None, :], axis=2
        )
        np.testing.assert_allclose(original_distances, transformed_distances)

    def test_least_squares_supports_different_dimensions(self):
        source = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        target = np.column_stack((source, source[:, 0] + source[:, 1]))
        alignment = SpaceAlignment.fit(
            source, target, "two_dim", "three_dim", method="least_squares"
        )
        np.testing.assert_allclose(alignment.transform(source), target, atol=1e-10)

    def test_soft_union_translates_query_and_keeps_relations_independent(self):
        source_before = self.source.vectors.copy()
        target_before = self.target.vectors.copy()
        results = self.manager.soft_union_query(
            [self.source, self.target],
            np.array([0.05, 0.02]),
            k=2,
            query_space="source_space",
        )
        self.assertEqual({result["source_relation"] for result in results},
                         {"source_db", "target_db"})
        self.assertTrue(all(result["id"] == "entity_0" for result in results))
        np.testing.assert_array_equal(self.source.vectors, source_before)
        np.testing.assert_array_equal(self.target.vectors, target_before)

    def test_hard_union_materialises_one_searchable_relation(self):
        union = self.manager.hard_union_integrate(
            [self.source, self.target], target_space="source_space"
        )
        self.assertIsInstance(union, UnifiedMetricRelation)
        self.assertEqual(union.vectors.shape, (8, 2))
        self.assertEqual(union.vector_space.model_name, "source_space")
        results = union.search(
            self.target_vectors[3], k=2, query_space="target_space"
        )
        self.assertEqual(
            set(results), {"source_db::entity_3", "target_db::entity_3"}
        )
        metadata = union.get_metadata(results)
        self.assertEqual(
            {item["source_relation"] for item in metadata},
            {"source_db", "target_db"},
        )

    def test_missing_alignment_is_reported(self):
        manager = UnionManager()
        with self.assertRaisesRegex(ValueError, "No alignment registered"):
            manager.soft_union_query(
                [self.source, self.target],
                self.source_vectors[0],
                query_space="source_space",
            )

if __name__ == "__main__":
    unittest.main()

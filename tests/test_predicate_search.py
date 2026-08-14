import unittest
import numpy as np
import os
import json
from main import SimpleMetricRelation
from core.base import BaseMetricIndex
from modules.predicate_search.engine import PredicateSearchModule

class MockIndex(BaseMetricIndex):
    """A minimal mock index to ensure Module 1 can be tested without Module 4."""
    def build(self, data): pass
    def search(self, query_vector, k=10): return []

class TestPredicateSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the generated data for testing if available, otherwise mock it
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
            # Fallback to a small mock if data doesn't exist
            cls.dim = 128
            cls.vectors = np.random.rand(10, cls.dim).astype(np.float32)
            cls.metadata = [{"id": f"item_{i}", "category": "test"} for i in range(10)]
            cls.relation = SimpleMetricRelation("MockRelation", cls.vectors, cls.metadata)

    def setUp(self):
        # [INDEPENDENCE] We use MockIndex here to demonstrate independence from Module 4.
        # Students: When you want to test with the REAL Module 4 implementation, 
        # replace MockIndex() with MetricIndex() from modules.metric_indexing.indexing.
        self.index = MockIndex()
        self.index.build(self.relation)
        self.module = PredicateSearchModule(self.index, self.relation)

    def test_hybrid_search_interface(self):
        """Verify that hybrid_search returns a list (even if empty for now)."""
        query_vector = np.random.rand(self.dim).astype(np.float32)
        filters = {"category": "electronics"}
        results = self.module.hybrid_search(query_vector, filters, k=5)
        self.assertIsInstance(results, list)

    def test_relation_integration(self):
        """Test if the module can correctly access the relation's metadata."""
        item_ids = [self.metadata[0]['id']]
        metadata = self.module.relation.get_metadata(item_ids)
        self.assertEqual(metadata[0]['id'], item_ids[0])

if __name__ == "__main__":
    unittest.main()

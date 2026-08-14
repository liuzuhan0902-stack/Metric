import unittest
import numpy as np
import os
import json
from main import SimpleMetricRelation
from modules.union_management.union import UnionManager

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
            # Should not crash even if it returns None/empty
            self.manager.soft_union_query([self.relation], query_vector)
        except Exception as e:
            self.fail(f"soft_union_query failed with {e}")

    def test_hard_union_interface(self):
        """Test if hard_union_integrate accepts a list of relations."""
        try:
            # Should return a MetricRelation or None (as it's a TODO)
            result = self.manager.hard_union_integrate([self.relation])
        except Exception as e:
            self.fail(f"hard_union_integrate failed with {e}")

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import os
import json
from main import SimpleMetricRelation
from modules.entity_linking.linker import EntityLinker

class TestEntityLinking(unittest.TestCase):
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
        self.linker = EntityLinker()

    def test_find_anchor_points_interface(self):
        """Test if find_anchor_points accepts two relations."""
        try:
            # We can use the same relation twice for a simple interface check
            anchors = self.linker.find_anchor_points(self.relation, self.relation)
            self.assertIsInstance(anchors, list)
        except Exception as e:
            self.fail(f"find_anchor_points failed with {e}")

    def test_link_entities_interface(self):
        """Test if link_entities accepts two relations."""
        try:
            self.linker.link_entities(self.relation, self.relation)
        except Exception as e:
            self.fail(f"link_entities failed with {e}")

if __name__ == "__main__":
    unittest.main()

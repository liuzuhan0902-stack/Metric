import unittest
import numpy as np
import os
import json
from main import SimpleMetricRelation
from modules.predicate_search.engine import PredicateSearchModule
from modules.query_engine.parser import SQLParser
from modules.union_management.union import UnionManager
from modules.metric_indexing.indexing import MetricIndex
from modules.entity_linking.linker import EntityLinker
from data.data_generator import generate_synthetic_data

class TestSystemIntegration(unittest.TestCase):
    """
    Integration tests to verify how all modules work together.
    This test suite demonstrates the end-to-end flow of the MetricDB system.
    """
    
    @classmethod
    def setUpClass(cls):
        # 1. Setup data
        cls.data_dir = "data"
        cls.dim = 64
        cls.size = 50
        generate_synthetic_data(num_samples=cls.size, dimension=cls.dim, output_dir=cls.data_dir)
        
        vectors = np.load(os.path.join(cls.data_dir, "vectors.npy"))
        with open(os.path.join(cls.data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        cls.relation = SimpleMetricRelation("IntegrationInventory", vectors, metadata)

    def test_full_workflow(self):
        """
        [INTEGRATION] Simulate a full workflow involving multiple modules.
        This test uses real module implementations (not mocks) to ensure 
        they can communicate correctly.
        
        Flow: Parsing (M2) -> Indexing (M4) -> Predicate Search (M1)
        """
        print("\nStarting full workflow integration test...")
        
        # Step 1: Parse a query (Module 2)
        parser = SQLParser()
        sql = "SELECT * FROM Inventory WHERE category = 'electronics' AND vector SIMILAR TO query"
        query_plan = parser.parse(sql)
        # For now, it might be None as it's a TODO, but we check the flow
        
        # Step 2: Build the index (Module 4)
        index = MetricIndex()
        index.build(self.relation)
        
        # Step 3: Initialize Predicate Search (Module 1)
        ps_module = PredicateSearchModule(index, self.relation)
        
        # Step 4: Perform search
        query_vec = np.random.rand(self.dim).astype(np.float32)
        results = ps_module.hybrid_search(query_vec, {"category": "electronics"}, k=5)
        
        # Verify result type
        self.assertIsInstance(results, list)
        print("Integration flow (Partial) completed successfully.")

    def test_cross_relation_workflow(self):
        """
        Simulate workflow involving multiple relations:
        Union Management (Module 3) and Entity Linking (Module 5)
        """
        print("\nStarting cross-relation integration test...")
        
        # Create a second relation for testing
        vectors_b = np.random.rand(self.size, self.dim).astype(np.float32)
        metadata_b = [{"id": f"external_{i}", "category": "misc"} for i in range(self.size)]
        relation_b = SimpleMetricRelation("ExternalDB", vectors_b, metadata_b)
        
        # Module 3: Union Management
        union_mgr = UnionManager()
        # Test soft union interface
        query_vec = np.random.rand(self.dim).astype(np.float32)
        union_mgr.soft_union_query([self.relation, relation_b], query_vec)
        
        # Module 5: Entity Linking
        linker = EntityLinker()
        anchors = linker.find_anchor_points(self.relation, relation_b)
        self.assertIsInstance(anchors, list)
        
        print("Cross-relation flow completed successfully.")

if __name__ == "__main__":
    unittest.main()

import sys
import os

# Add the project root to sys.path to allow running this script from the examples folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import MetricRelation
from typing import List, Dict, Any
import numpy as np

class InMemoryMetricRelation(MetricRelation):
    """
    A simple in-memory implementation of the MetricRelation interface.
    This class is intended for demonstration purposes and basic testing.
    It stores vectors and metadata in dictionaries indexed by entity IDs.
    """
    def __init__(self, data: Dict[str, np.ndarray], metadata: Dict[str, Dict[str, Any]]):
        """
        Initialize the relation with data and metadata.

        Args:
            data: A dictionary mapping IDs to NumPy vector arrays.
            metadata: A dictionary mapping IDs to metadata dictionaries.
        """
        self.data = data
        self.metadata = metadata

    def get_vectors(self, ids: List[str]) -> np.ndarray:
        """
        Retrieve vector embeddings for the specified IDs.

        Args:
            ids: A list of unique identifiers for the items to retrieve.

        Returns:
            A NumPy array containing the requested vectors.
        """
        return np.array([self.data[i] for i in ids])

    def get_metadata(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve metadata dictionaries for the specified IDs.

        Args:
            ids: A list of unique identifiers.

        Returns:
            A list of dictionaries containing metadata for each ID.
        """
        return [self.metadata[i] for i in ids]

def run_example():
    """
    A standalone function that demonstrates the basic usage of the MetricRelation interface.
    This shows how a student might create a simplified version of the system for 
    prototyping their module's logic.
    """
    print("=== MetricDB: Basic Usage Example ===")
    print("Initializing example MetricDB components...")
    
    # 1. Prepare sample data
    # In a real scenario, this data would be loaded from disk or generated via data_generator.py.
    ids = ["v1", "v2", "v3"]
    vectors = {
        "v1": np.array([0.1, 0.2], dtype=np.float32),
        "v2": np.array([0.5, 0.6], dtype=np.float32),
        "v3": np.array([0.9, 0.1], dtype=np.float32)
    }
    metadata = {
        "v1": {"label": "cat", "price": 10},
        "v2": {"label": "dog", "price": 20},
        "v3": {"label": "bird", "price": 15}
    }
    
    # 2. Instantiate the relation
    # MetricRelation is the core data abstraction in the system.
    relation = InMemoryMetricRelation(vectors, metadata)
    print(f"Created InMemoryMetricRelation with {len(ids)} items.")
    
    # 3. Demonstrate vector retrieval
    # Students can use this to get raw vector data for similarity calculations.
    target_id = "v1"
    v = relation.get_vectors([target_id])
    print(f"Retrieved vector for {target_id}: {v}")
    
    # 4. Demonstrate metadata retrieval
    # Useful for Module 1 (Predicate Search) to filter results based on attributes.
    meta = relation.get_metadata([target_id])
    print(f"Retrieved metadata for {target_id}: {meta}")
    
    print("\nExample finished successfully.")

if __name__ == "__main__":
    run_example()

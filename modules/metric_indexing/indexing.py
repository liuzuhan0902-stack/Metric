from core.base import BaseMetricIndex, MetricRelation
import numpy as np

class MetricIndex(BaseMetricIndex):
    """
    Module 4: From Vector Indexing to Metric Indexing (Assigned to: Student 4)
    
    Objective: Develop index structures that are decoupled from specific 
    embedding coordinates, focusing instead on the metric properties (distances).
    
    Task for Student 4:
    Extend or implement indexing structures (like M-Tree, Vantage Point Tree, etc.)
    that can build and search over MetricRelation data.
    """
    def build(self, data: MetricRelation):
        """
        Build the metric index.
        
        Args:
            data: The MetricRelation containing the vectors to be indexed.
        """
        # TODO [Student 4]: Build metric index (from scratch or by wrapping existing indices like FAISS)
        # Hint: In MetricDB, the index should be aware of the distance metric used.
        pass

    def search(self, query_vector: np.ndarray, k: int = 10):
        """
        Perform a metric-based search.
        
        Args:
            query_vector: The query point.
            k: Number of neighbors to return.
        """
        # TODO [Student 4]: Implement metric search logic (e.g., triangular inequality pruning)
        return []

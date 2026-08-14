import os
import tempfile
import unittest

import numpy as np

from experiments.union_benchmark import (
    generate_multispace_data,
    run_configuration,
    write_results,
)


class TestUnionBenchmark(unittest.TestCase):
    def test_generated_spaces_share_latent_structure(self):
        dataset = generate_multispace_data(3, 12, 4, 3, seed=11)
        self.assertEqual(len(dataset.relations), 3)
        latent_distances = np.linalg.norm(
            dataset.latent_vectors[1][0] - dataset.latent_vectors[1][1]
        )
        observed_distances = np.linalg.norm(
            dataset.relations[1].vectors[0] - dataset.relations[1].vectors[1]
        )
        self.assertAlmostEqual(latent_distances, observed_distances)

    def test_benchmark_executes_all_real_search_paths(self):
        rows = run_configuration(3, 20, 4, query_count=5, k=3, seed=17)
        self.assertEqual(len(rows), 5)
        keyed = {(row["method"], row["alignment_state"]): row for row in rows}
        self.assertIn(("single", "not_applicable"), keyed)
        self.assertIn(("soft", "before"), keyed)
        self.assertIn(("soft", "after"), keyed)
        self.assertIn(("hard", "before"), keyed)
        self.assertIn(("hard", "after"), keyed)
        self.assertLess(
            keyed[("soft", "after")]["alignment_rmse"],
            keyed[("soft", "before")]["alignment_rmse"],
        )
        self.assertEqual(keyed[("soft", "after")]["recall_at_k"], 1.0)
        self.assertEqual(keyed[("hard", "after")]["recall_at_k"], 1.0)
        self.assertGreaterEqual(keyed[("hard", "after")]["build_latency_ms"], 0.0)

    def test_results_are_written_as_csv(self):
        rows = run_configuration(2, 10, 3, query_count=2, k=2, seed=23)
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "results.csv")
            write_results(rows, output)
            with open(output, encoding="utf-8") as result_file:
                contents = result_file.read()
        self.assertIn("query_latency_ms", contents)
        self.assertIn("recall_at_k", contents)
        self.assertIn("alignment_rmse", contents)


if __name__ == "__main__":
    unittest.main()

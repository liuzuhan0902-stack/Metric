# Virtual and Hard Union Benchmark

`union_benchmark.py` compares three real query paths:

1. **Single** — exact search over the first database only.
2. **Soft** — `UnionManager.soft_union_query` translates the query and searches
   every independent database.
3. **Hard** — `UnionManager.hard_union_integrate` performs the offline materialisation,
   followed by online `UnifiedMetricRelation.search` calls over one vector matrix.

## Dataset methodology

A seeded latent semantic space is sampled first. Each database receives a disjoint
batch of latent records. Independent orthogonal transforms and translations then
produce the observed embedding spaces. Orthogonal transforms preserve pairwise
Euclidean structure, while translations and rotations make raw coordinates
incompatible. A separate shared anchor set is used to learn mappings and is not
included among searchable records.

The default matrix covers 2, 3, and 5 databases; 100 and 500 vectors per database;
and 16- and 32-dimensional embeddings. Each configuration executes 20 queries at
`k=10` with seed 2026.

## Run

```bash
python experiments/union_benchmark.py
```

Results are written to `experiments/results/union_benchmark.csv`. A smaller custom
run can be executed with:

```bash
python experiments/union_benchmark.py \
  --databases 2,3 --vectors 50,100 --dimensions 8,16 \
  --queries 10 --k 5 --seed 2026 --output experiments/results/custom.csv
```

## CSV columns

- `database_count`, `vectors_per_database`, `total_vectors`, `dimension`: scale.
- `query_count`, `k`, `seed`: reproducibility parameters.
- `method`: `single`, `soft`, or `hard`.
- `alignment_state`: `before`, `after`, or `not_applicable`.
- `query_latency_ms`: mean online latency per query.
- `build_latency_ms`: hard-union offline materialisation time; zero otherwise.
- `recall_at_k`: mean overlap with global latent-space exact top-k.
- `alignment_rmse`: anchor RMSE against Space A before or after transformation.

Latency results should be rerun on the hardware described in the dissertation;
the committed CSV is a reproducible functional result from the development
environment, not a hardware-independent performance claim.

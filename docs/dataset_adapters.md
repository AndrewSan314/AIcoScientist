# Dataset adapters

`DatasetSpec` describes the model and candidate schema, provenance, and data leakage boundaries. A `DatasetAdapter` owns
loading, feature construction, candidate generation, optional constraints, and
dataset-specific recommendation formatting. The registry currently exposes only
`si_mxene`; additional adapters can load directly into DataFrames.

## Leakage Prevention & Visibility Semantics

The pipeline enforces three explicit visibility tiers:
- **Known Before Experiment (Candidates)**: Design features and parameters known prior to running an experiment.
- **Revealed After Query (Observations)**: Explicitly revealed experimental feedback (`observation_columns`) and target.
- **Hidden From Model (Oracle / Ground Truth)**: Ground truth data (`oracle_columns`, hidden post-outcome metrics, future cycles) accessible ONLY to the offline oracle and never exposed during model training or replay.

### Key Semantics in `DatasetSpec`
- `entity_id_column`: Identifies the physical experimental entity (e.g., a battery cell).
- `candidate_id_column`: Identifies the optimization candidate (e.g., cycling protocol).
- `split_group_columns`: Columns whose groups must never be split across train and test partitions (enforced via `GroupShuffleSplit`).
- `oracle_columns`: Hidden ground-truth columns strictly forbidden from appearing in model features.
- `observation_columns`: Allowed post-query observation columns returned to the observer/replay loop.
- `feature_horizon`: Cutoff metadata for cycle/time-series truncation.
- `source_dataset` / `source_version`: Provenance tracking metadata.

### `DatasetBundle`
A clean dataclass container representing data visibility boundaries:
- `candidates`: Pre-experiment candidates DataFrame.
- `observations`: Observed training data DataFrame.
- `oracle`: Hidden offline evaluation DataFrame.
- `provenance`: Metadata dictionary.

The Si/MXene adapter is the only adapter coupled to `experiment_store.py` and
`chemistry_rules.py`. The SQLite schema remains intentionally Si/MXene-specific
for this refactor; generic benchmark adapters must not depend on that store.

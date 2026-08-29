# Dataset adapters

`DatasetSpec` describes the model and candidate schema, provenance, and data leakage boundaries. A `DatasetAdapter` owns
loading, feature construction, candidate generation, optional constraints, and
dataset-specific recommendation formatting. The registry currently exposes only
`si_mxene`; additional adapters can load directly into DataFrames.

## Identity Concepts & Distinction

To prevent confounding physical samples with optimization search space variables, three distinct identity concepts are supported:

1. **`entity_id_column` (Physical Entity Identity)**:
   - Identifies the physical experimental specimen or hardware entity (e.g., a specific physical battery cell `cell_001` or coin cell sample).
   - Multiple time steps, cycles, or measurements can share the same `entity_id_column`.
   - Used for group-safe cross-validation and preventing physical cell leakage across train/test partitions.

2. **`candidate_id_column` (Candidate / Protocol Identity)**:
   - Identifies the optimization protocol, candidate recipe, or formulation identifier (e.g., protocol name `P17` or recipe name `Si-MX-04`).
   - Multiple physical cells (`cell_001`, `cell_002`) can test the same candidate protocol (replicates).

3. **`candidate_columns` (Design Coordinates / Variables)**:
   - Numerical or categorical parameter coordinates that define the candidate in the surrogate model search space (e.g., `["si_content", "mxene_content", "drying_temp"]` or `["charging_c_rate", "cutoff_voltage"]`).
   - Used by the model and acquisition function to determine parameter coordinates and distances.

## Leakage Prevention & Visibility Semantics

The pipeline enforces three explicit visibility tiers:
- **Known Before Experiment (Candidates)**: Design features and parameters known prior to running an experiment (`candidate_columns`, static defaults). Must never contain post-outcome information or oracle fields.
- **Revealed After Query (Observations)**: Explicitly revealed experimental feedback (`observation_columns`) and target. Cannot overlap with hidden oracle columns.
- **Hidden From Model (Oracle / Ground Truth)**: Ground truth data (`oracle_columns`, hidden post-outcome metrics, future cycles) accessible ONLY to the offline oracle and never exposed during model training or replay.

### Key Semantics in `DatasetSpec`
- `entity_id_column`: Identifies the physical experimental entity (non-empty string when provided).
- `candidate_id_column`: Identifies the optimization candidate / protocol (non-empty string when provided).
- `split_group_columns`: Columns whose groups must never be split across train and test partitions (enforced via `GroupShuffleSplit`).
- `oracle_columns`: Hidden ground-truth columns strictly forbidden from appearing in `feature_columns`, `candidate_columns`, or `observation_columns`.
- `observation_columns`: Allowed post-query observation columns returned to the observer/replay loop.
- `feature_horizon`: Positive integer cutoff for cycle/time-series truncation.
- `source_dataset` / `source_version`: Provenance tracking metadata.

### Replicate & Observation Rules in `OfflineOracle`
- `replicate_policy="error"`: Fails immediately if multiple rows match the same candidate coordinates.
- `replicate_policy="mean"`: Aggregates target values across replicates, computing mean target and `target_std`.
  - **Safety Guard**: If `observation_columns` are present on replicated candidates, querying raises `ValueError` to prevent silently mixing mean targets with observations from an arbitrary single replicate.

### `DatasetBundle`
A clean dataclass container representing data visibility boundaries:
- `candidates`: Pre-experiment candidates DataFrame.
- `observations`: Observed training data DataFrame.
- `oracle`: Hidden offline evaluation DataFrame.
- `provenance`: Metadata dictionary.

The Si/MXene adapter is the only adapter coupled to `experiment_store.py` and
`chemistry_rules.py`. The SQLite schema remains intentionally Si/MXene-specific
for this refactor; generic benchmark adapters must not depend on that store.

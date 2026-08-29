# Dataset adapters

`DatasetSpec` describes the model and candidate schema. A `DatasetAdapter` owns
loading, feature construction, candidate generation, optional constraints, and
dataset-specific recommendation formatting. The registry currently exposes only
`si_mxene`; additional adapters can load directly into DataFrames.

The Si/MXene adapter is the only adapter coupled to `experiment_store.py` and
`chemistry_rules.py`. The SQLite schema remains intentionally Si/MXene-specific
for this refactor; generic benchmark adapters must not depend on that store.

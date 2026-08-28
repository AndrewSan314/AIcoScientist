# Production Data Foundation Design

## Understanding summary

- The existing MVP already runs CSV ingestion, model training, recommendation, and Streamlit end to end.
- Phase 1 upgrades only the data foundation; model, recommendation, dashboard, authentication, and closed-loop automation stay unchanged.
- SQLite becomes the validated source of truth while CSV remains the import/export format.
- Every ingestion is atomic and traceable to source files by SHA-256.
- Invalid data must stop the pipeline before training.
- The phase is complete only when focused tests and the full existing suite pass.

## Assumptions

- Initial scale is below 10,000 experiments and one local writer is sufficient.
- The application runs locally/on-premise and does not send proprietary data externally.
- High availability, multi-user access, and authentication are deferred.
- Python's standard `sqlite3` module is sufficient; no ORM or new dependency is required.
- Synthetic input remains available for the demo, but the database records the exact imported sources.

## Architecture

`src/experiment_store.py` owns SQLite schema creation, CSV validation, transactional ingestion, source hashing, and loading validated source tables. The database lives at `data/experiments.db`. `src/build_dataset.py` delegates source ingestion/loading to this module and retains feature engineering plus master CSV export.

Tables:

- `schema_meta`: current schema version.
- `ingestion_runs`: run timestamps, status, row count, and error.
- `source_files`: dataset name, path, SHA-256, and run reference.
- `process_data`, `sem_features`, `edx_data`, `electrochem_data`: validated snapshots keyed by `sample_id`.

Each ingest validates all four DataFrames before opening the write transaction. The transaction replaces the four snapshot tables, inserts source lineage, and marks the run successful. Any SQLite failure rolls back all changes. Validation errors are recorded as failed runs without replacing the last valid snapshot.

## Testing strategy

- Valid CSVs create the database, four source tables, a successful run, and four source hashes.
- Invalid cross-table sample IDs fail without replacing the last valid snapshot.
- Stored hashes equal independently computed SHA-256 values.
- `build_master_dataset()` reads the validated SQLite snapshot.
- The existing end-to-end tests remain green.

## Decision log

| Decision | Alternatives | Reason |
|---|---|---|
| SQLite source of truth | JSON manifests, Pandera/Pydantic only | Transactions, constraints, and a base for a later experiment registry without an external service. |
| Standard `sqlite3` | ORM | The schema is small and an ORM adds no Phase 1 value. |
| Snapshot replacement | Append-only experiment history | Preserves current MVP behavior; versioned experiments belong to the closed-loop phase. |
| CSV import/export retained | Immediate instrument APIs | Keeps compatibility while production ingestion interfaces remain future work. |

## Known risks

- SQLite permits one writer at a time; move to PostgreSQL only when concurrent writers become necessary.
- Snapshot replacement does not represent multiple experimental rounds; a later phase will add append-only experiment versions.
- Scientific validity still depends on real experimental data and remains outside this software phase.

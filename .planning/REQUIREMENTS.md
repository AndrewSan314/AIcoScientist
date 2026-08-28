# Requirements: Battery AI Co-Scientist MVP

**Defined:** 2026-07-09
**Core Value:** Run end-to-end from sample CSV data to top-3 next fabrication recommendations.

## v1 Requirements

### Data

- [x] **DATA-01**: User can run the pipeline without real lab data and get synthetic raw CSVs with the expected schema.
- [x] **DATA-02**: User can build `data/processed/master_dataset.csv` by merging process, SEM, EDX, and electrochemical CSVs on `sample_id`.
- [x] **DATA-03**: User can inspect a `data_dictionary.md` that explains each input/output column.

### Model

- [x] **MODL-01**: User can train a RandomForest regression model to predict `retention_100`.
- [x] **MODL-02**: User receives `outputs/trained_model.pkl`, `outputs/model_metrics.json`, and `outputs/feature_importance.csv`.
- [x] **MODL-03**: User receives a Gaussian Process surrogate and scaler in the model bundle for uncertainty-aware recommendation.

### Recommendation

- [x] **RECO-01**: User can generate exactly top-3 candidate fabrication recipes.
- [x] **RECO-02**: Recommendations avoid recipes already present in the master dataset.
- [x] **RECO-03**: Each recommendation includes predicted retention, confidence, and a short reason.
- [x] **RECO-04**: Recommendations include uncertainty and UCB acquisition score over the discrete recipe grid.

### Interface

- [x] **UI-01**: User can open a Streamlit dashboard with dataset overview, model metrics, feature importance, and recommendations.
- [x] **UI-02**: Dashboard shows placeholder Accept, Modify, and Reject actions for each recommendation.

### Operations

- [x] **OPS-01**: User can run the full MVP with `python run_pipeline.py`.
- [x] **OPS-02**: User can run `pytest tests/` to verify dataset, model, and recommendation outputs.

## v2 Requirements

### Research

- [x] **RSCH-01**: Add `literature_notes.md` using the paper map from the source plan.
- **RSCH-02**: Excluded by user request: no LLM implementation.

### Optimization

- **OPTM-01**: Deferred: the source plan and ML upgrade explicitly limit this release to a local GP/UCB MVP.
- **OPTM-02**: Deferred: the source plan requires UI placeholders, not a database-backed experiment loop.

### Imaging

- [x] **IMAG-01**: Add SEM image feature extraction from real images.
- [x] **IMAG-02**: Add lightweight threshold-segmentation morphology features.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Authentication | Local demo only |
| Database backend | CSV files are enough for MVP |
| Scientific model validation | Synthetic data cannot validate scientific accuracy |
| Full closed-loop automation | Requires real experiment integration |

## Production Hardening Requirements

### Phase 1 - Data Foundation

- [x] **PDAT-01**: Validate schemas, numeric domains, unique IDs, matching sample sets, and formulation totals before training.
- [x] **PDAT-02**: Store the four validated source tables in SQLite in one atomic snapshot transaction.
- [x] **PDAT-03**: Record ingestion status, schema version, source path, row count, and SHA-256 for every import.
- [x] **PDAT-04**: Preserve the last valid snapshot and record a failed run when validation or ingestion fails.

### Phase 2 - Versioned Experiment Registry

- [x] **PREG-01**: Migrate schema version 1 to version 2 without losing scientific rows or source lineage.
- [x] **PREG-02**: Preserve multiple experiment batches using `(experiment_id, sample_id)` identities.
- [x] **PREG-03**: Make repeated imports of the same batch and source hashes idempotent.
- [x] **PREG-04**: Load the active experiment by default and allow explicit historical reads.
- [x] **PREG-05**: Keep the active experiment unchanged when a later import fails validation or transaction checks.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 1 | Complete |
| MODL-01 | Phase 3 | Complete |
| MODL-02 | Phase 3 | Complete |
| MODL-03 | Phase 3 | Complete |
| RECO-01 | Phase 4 | Complete |
| RECO-02 | Phase 4 | Complete |
| RECO-03 | Phase 4 | Complete |
| RECO-04 | Phase 4 | Complete |
| UI-01 | Phase 6 | Complete |
| UI-02 | Phase 6 | Complete |
| OPS-01 | Phase 7 | Complete |
| OPS-02 | Phase 8 | Complete |
| PDAT-01 | Production Phase 1 | Complete |
| PDAT-02 | Production Phase 1 | Complete |
| PDAT-03 | Production Phase 1 | Complete |
| PDAT-04 | Production Phase 1 | Complete |
| PREG-01 | Production Phase 2 | Complete |
| PREG-02 | Production Phase 2 | Complete |
| PREG-03 | Production Phase 2 | Complete |
| PREG-04 | Production Phase 2 | Complete |
| PREG-05 | Production Phase 2 | Complete |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-07-09*
*Last updated: 2026-07-15 after Production Phase 2*

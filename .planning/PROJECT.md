# Battery AI Co-Scientist MVP

## What This Is

A runnable AI co-scientist workflow that integrates fabrication parameters, SEM-derived features, EDX composition data, and electrochemical results for nano-silicon/few-layer Ti3C2Tx MXene anodes using sodium alginate binder. The model remains a proposal/demo until real experimental validation, while production hardening now proceeds one tested phase at a time.

## Core Value

The project must run end-to-end from sample CSV data to top-3 next fabrication recommendations.

## Requirements

### Validated

- [x] Build synthetic/sample raw CSVs with the same schema expected for real experiments.
- [x] Merge raw CSVs into one master dataset with engineered features.
- [x] Train RandomForest and Gaussian Process models for `retention_100`.
- [x] Generate top-3 recipes with GP uncertainty, UCB score, confidence, and reason.
- [x] Extract lightweight morphology features from public battery SEM images.
- [x] Show dataset, metrics, feature importance, recommendations, and decision placeholders in Streamlit.
- [x] Provide one command to run the full pipeline.
- [x] Provide literature notes, a one-minute demo script, and two presentation screenshots.
- [x] Store validated scientific inputs in SQLite with atomic ingestion, schema versioning, and SHA-256 source lineage (Production Phase 1).
- [x] Preserve versioned experimental rounds with idempotent imports, active-dataset reads, and a lossless schema v1-to-v2 migration (Production Phase 2).

### Out of Scope

- LLM literature mining - future data curation module.
- SAM/SAM2 deep segmentation - threshold segmentation is sufficient for the MVP.
- Research-grade closed-loop Bayesian Optimization - future recommender.
- Multi-user database service and authentication - deferred until local SQLite operation is insufficient.
- Scientific accuracy claims from synthetic data - real experimental data is required.

## Context

The source plan is `AI_CoScientist_Code_Pipeline_Phases_and_Papers.md`. Legacy ZnSe/SnSe battery prediction files were removed so the repo focuses on the Si/MXene AI co-scientist MVP.

## Constraints

- **Tech stack**: Python, pandas, scikit-learn, joblib, Streamlit, pytest - small local demo stack.
- **Data**: Synthetic/sample data is acceptable for MVP, but schema must match future real lab data.
- **Target**: `retention_100` is the primary target; no target leakage in recommendation-time features.
- **Scope**: Do not implement heavy research modules before the core data -> model -> recommendation -> dashboard path works.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Remove old ZnSe/SnSe files | They were a different domain and cluttered the MVP codebase | Good |
| Generate synthetic CSVs when missing | Makes `python run_pipeline.py` runnable from a clean checkout | Good |
| Use RandomForestRegressor | Simple, installed dependency, feature importance available | Good |
| Add scikit-learn GaussianProcessRegressor + UCB | Provides uncertainty-aware recipe ranking without adding heavy BO dependencies | Good |
| Use mean-filled SEM/EDX estimates for recommendations | Candidate recipes do not have real SEM/EDX before fabrication | Good |
| Require 100 wt% composition and at least 5 wt% carbon | Prevents chemically invalid synthetic and recommended recipes | Good |
| Use standard-library SQLite for Production Phase 1 | Adds transactions, constraints, and lineage without an ORM or external service | Good |
| Replace the validated SQLite snapshot per ingest in Phase 1 | Preserved MVP behavior while the schema stabilized | Superseded by the Phase 2 append-only registry |
| Identify experiments by batch ID plus source hashes | Makes retries idempotent while allowing identical measurements from distinct lab batches | Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? Move to Out of Scope with reason.
2. Requirements validated? Move to Validated with phase reference.
3. New requirements emerged? Add to Active.
4. Decisions to log? Add to Key Decisions.
5. "What This Is" still accurate? Update if drifted.

**After each milestone**:
1. Full review of all sections.
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state.

---
*Last updated: 2026-07-15 after Production Phase 2*

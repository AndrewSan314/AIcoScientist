# AI Co-Scientist Production Roadmap

> **For implementation agents:** Create and approve a dedicated implementation plan for the selected phase, then use `executing-plans` and TDD. Never implement more than one phase per user-approved iteration.

**Goal:** Evolve the current Battery AI Co-Scientist demo into a traceable, recoverable, testable production system without weakening scientific safeguards.

**Architecture:** Harden the existing Python/SQLite pipeline incrementally. Preserve stable DataFrame and artifact contracts while adding versioned experiment history, operational runs, model/recommendation registries, human decisions, closed-loop evaluation, characterization QC, optional cited reasoning, and final deployment controls.

**Current stack:** Python, SQLite, pandas, scikit-learn, Streamlit, pytest

**Source documents:** `AI_FULL_PIPELINE.md`, `AI_CoScientist_Code_Pipeline_Phases_and_Papers.md`, `.planning/PROJECT.md`

---

## Execution rules

1. Implement exactly one production phase at a time.
2. Confirm the phase design and explicit non-goals before editing code.
3. Run Semble and GitNexus impact analysis before modifying indexed symbols.
4. Use RED-GREEN-REFACTOR for every behavior change.
5. A phase cannot complete until its focused tests, the full regression suite, compilation, dependency checks, and end-to-end pipeline pass.
6. Database phases require migration, rollback, integrity, and foreign-key checks.
7. Update this roadmap, `.planning/STATE.md`, `.planning/PROJECT.md`, requirements, and operator documentation at each phase boundary.
8. Do not begin the next phase without a new explicit user instruction.

## Status overview

| Phase | Name | Status | Depends on |
|---|---|---|---|
| 1 | SQLite Data Foundation | Complete | MVP |
| 2 | Versioned Experiment Registry | Complete | Phase 1 |
| 3 | Operational Pipeline and Run Registry | Next | Phase 2 |
| 4 | Model Registry and Reproducibility | Planned | Phase 3 |
| 5 | Production Recommendation Engine | Planned | Phase 4 |
| 6 | Human Review and Experiment Planning | Planned | Phase 5 |
| 7 | Closed-Loop Learning and Evaluation | Planned | Phase 6 |
| 8 | Characterization Pipeline Productionization | Planned | Phase 7 |
| 9 | Cited AI Scientist Reasoning | Optional | Phases 5-8 |
| 10 | Security, Recovery, and Release Readiness | Planned | All enabled phases |

## Phase 1 - SQLite Data Foundation

**Status:** Complete on 2026-07-15.

**Delivered:**

- Schema-versioned SQLite store at `data/experiments.db`.
- Strict validation of schemas, IDs, numeric domains, and formulation totals.
- Atomic snapshot replacement with rollback.
- Ingestion runs and SHA-256 source-file lineage.
- Master dataset builder routed through validated SQLite data.

**Verified gate:** 8 tests passed; compile, dependency, pipeline, SQLite integrity, and foreign-key checks passed.

**Resolved ceiling:** Phase 2 replaced snapshot-only storage with versioned experiment history.

## Phase 2 - Versioned Experiment Registry

**Status:** Complete on 2026-07-15.

**Goal:** Preserve every experimental round and sample instead of replacing the previous snapshot.

**Deliverables:**

- SQLite schema version 2 with stable `experiment_id`, `sample_id`, batch, operator, timestamps, and source-run relationships.
- Append-only experiment and measurement history for process, SEM, EDX, and electrochemistry records.
- Idempotent import keyed by source hash and logical experiment identity.
- An active-dataset query that preserves the current model-ready DataFrame contract.
- Migration from schema version 1 without data loss.

**Phase gate:**

- Migration preserves every Phase 1 row and lineage record.
- Two experimental rounds coexist and remain independently queryable.
- Re-importing identical sources does not duplicate experiments.
- Mid-transaction failure leaves no partial experiment.
- Model and recommendation tests pass against the active-dataset query.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Multi-user authentication, PostgreSQL, dashboard decisions, automatic retraining.

**Verified gate:** Lossless v1-to-v2 migration, multi-batch history, idempotent retries, rollback safety, active/historical reads, downstream model compatibility, SQLite integrity, and end-to-end execution passed.

## Phase 3 - Operational Pipeline and Run Registry

**Goal:** Make pipeline execution observable, restartable, and safe for operators.

**Deliverables:**

- Explicit commands: `ingest`, `train`, `recommend`, `report`, and `run-all`.
- Pipeline run records with status, timestamps, inputs, outputs, and failure details.
- Structured logging and deterministic exit codes.
- Configuration separation for demo and real-data modes; production mode never silently generates synthetic data.
- Safe retry/resume for completed stages without corrupting prior artifacts.

**Phase gate:**

- Every command has success and failure-path tests.
- Interrupted runs are marked failed and can restart safely.
- Real-data mode fails clearly when required inputs are absent.
- Re-running a completed stage is deterministic or explicitly versioned.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Distributed scheduler, message queue, Kubernetes deployment.

## Phase 4 - Model Registry and Reproducibility

**Goal:** Make every trained model reproducible and traceable to exact data and features.

**Deliverables:**

- Model-run records linked to experiment snapshot, data hash, feature schema, code/config version, metrics, and artifacts.
- Immutable model versions and a controlled active-model pointer.
- Machine-readable model card covering intended use, limitations, and synthetic/real-data status.
- Target-leakage checks and deterministic training seeds.
- Initial drift and uncertainty-calibration reports when sufficient real data exists.

**Phase gate:**

- Same data/config/seed reproduces equivalent outputs within declared tolerance.
- Missing or changed features fail before training/prediction.
- Artifact/hash mismatch is detected.
- Active-model promotion and rollback are tested.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Claiming scientific validity from synthetic data, introducing heavier models without evidence.

## Phase 5 - Production Recommendation Engine

**Goal:** Persist feasible, reproducible recommendations tied to a specific model and lab constraints.

**Deliverables:**

- Versioned recommendation records with model ID, candidate space, constraints, acquisition settings, and evidence.
- Hard constraint validation for formulation totals, equipment ranges, safety limits, and previously tested recipes.
- Diverse batch recommendations rather than near-duplicate top scores.
- Configurable experiment budget and optional multi-objective scoring for retention, capacity, CE, Rct, cost, and risk.
- Stable report schema for downstream UI and reasoning.

**Phase gate:**

- No infeasible or previously tested recipe can pass hard constraints.
- Batch diversity and budget limits are deterministic and tested.
- Every recommendation resolves to its model, data snapshot, and configuration.
- Report-schema compatibility tests pass.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Autonomous lab execution and unsupported scientific mechanism claims.

## Phase 6 - Human Review and Experiment Planning

**Goal:** Replace dashboard placeholders with persisted, auditable scientist decisions.

**Deliverables:**

- Accept, modify, and reject decisions linked to recommendation and operator.
- Original and modified recipes retained with reasons and timestamps.
- Modified recipes revalidated and rescored before acceptance.
- Accepted recommendations create versioned experiment plans and exportable lab protocols.
- Immutable decision audit trail.

**Phase gate:**

- Each decision transition and invalid transition is tested.
- Modified recipes cannot bypass Phase 5 constraints.
- Refresh/restart does not lose decisions.
- Audit history cannot be silently overwritten.
- Full UI, regression, and end-to-end checks pass.

**Explicit non-goals:** Role-based access control and automated instrument control.

## Phase 7 - Closed-Loop Learning and Evaluation

**Goal:** Ingest results from accepted experiments and measure whether recommendations improve outcomes.

**Deliverables:**

- Accepted plan to executed experiment to measured-result linkage.
- Controlled retrain/re-rank workflow after validated new results arrive.
- Prediction-versus-actual history and per-round error reporting.
- Uncertainty calibration, drift, top-N hit rate, and recommendation acceptance metrics.
- Manual approval gate before promoting a newly trained model.

**Phase gate:**

- A complete simulated round trip is reproducible in tests.
- Failed measurements remain recorded but cannot enter training data.
- Retraining uses only approved, validated experiment versions.
- Model quality regressions block promotion.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Fully autonomous retraining or unattended scientific decisions.

## Phase 8 - Characterization Pipeline Productionization

**Goal:** Make SEM/EDX-derived features traceable, quality-controlled, and suitable for real experiments.

**Deliverables:**

- Raw image/artifact registry with instrument, magnification, scale, operator, and calibration metadata.
- Versioned masks, feature-extractor version, parameters, and QC status.
- Human QC/override workflow for segmentation failures.
- Domain-shift checks between demo and Si/MXene images.
- Training/fine-tuning workflow only when representative labeled data is available.

**Phase gate:**

- Raw image to mask to feature row lineage is complete.
- Corrupt, uncalibrated, and unsupported images fail safely.
- Re-extraction never overwrites historical masks/features.
- Repeatability is tested on fixed reference images.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Claiming SAM/SAM2 scientific accuracy without labeled domain validation.

## Phase 9 - Cited AI Scientist Reasoning

**Status:** Optional; enable only when a trusted document corpus and an approved model boundary exist.

**Goal:** Generate structured scientific explanations grounded in stored evidence and citations.

**Deliverables:**

- Versioned retrieval corpus for literature, SOPs, experiment notes, and failed experiments.
- Structured reasoning schema containing recommendation, comparison, risks, suggested measurements, and citations.
- Guardrails requiring numeric claims to resolve to database/model artifacts and citations to resolve to retrieved sources.
- Explicit `insufficient evidence` behavior.
- Prompt/model/version, latency, and cost tracking when an external model is used.

**Phase gate:**

- Citation resolution and unsupported-claim rejection tests pass.
- Prompt-injection and proprietary-data boundary tests pass.
- Invalid structured outputs are rejected rather than rendered.
- The core pipeline remains usable with reasoning disabled.
- Full regression and end-to-end checks pass.

**Explicit non-goals:** Letting an LLM fabricate measurements, modify source data, or autonomously approve experiments.

## Phase 10 - Security, Recovery, and Release Readiness

**Goal:** Validate the enabled system as an operable production release.

**Deliverables:**

- Authentication and least-privilege roles if deployment becomes multi-user.
- Input/upload hardening, secrets management, and external-data egress controls.
- Database/artifact backup, restore, retention, and migration runbooks.
- Health checks, audit logs, operational metrics, and alertable failure states.
- CI gates for tests, schema migration checks, dependency health, and reproducible packaging.
- Deployment and rollback documentation for the selected environment.

**Phase gate:**

- Backup restoration reproduces a usable experiment/model/recommendation chain.
- Authorization and audit tests cover all enabled write paths.
- Clean-environment install and smoke tests pass.
- Failure-recovery drill and rollback pass.
- All enabled phase gates pass together.

## Production completion criteria

The application may be called production-ready only when:

- Real laboratory data can be ingested without synthetic fallback.
- Every sample is traceable from raw source through features, training data, model, recommendation, decision, and measured result.
- Models and recommendations are versioned, reproducible, and recoverable.
- Scientist decisions persist and remain auditable.
- New results can enter a controlled retrain/re-rank loop.
- Prediction-versus-actual and calibration metrics are visible.
- Enabled AI reasoning is cited, schema-valid, and safely disableable.
- Backup/restore, security boundaries, clean deployment, and rollback are verified.
- Scientific claims are limited to evidence supported by real validation data.

## Decision log

| Decision | Reason |
|---|---|
| Keep SQLite until concurrent writers or deployment topology require PostgreSQL | Avoid infrastructure before measured need while preserving migration paths. |
| Build historical experiment identity before operational/model registries | Downstream reproducibility requires stable experiment versions. |
| Persist models before productionizing recommendations | Recommendations must identify the exact model and data that produced them. |
| Persist human decisions before enabling closed-loop retraining | Feedback must be auditable and scientist-controlled. |
| Place characterization hardening after the structured closed loop | Domain image work requires real samples, metadata, and feedback infrastructure. |
| Keep LLM/RAG optional and late | Reasoning is unsafe and hard to evaluate before trusted evidence, citations, and guardrails exist. |
| Make release hardening the final cross-phase gate | Security and recovery must cover the actual enabled surface, not speculative components. |

---

*Created: 2026-07-15. Update only at an approved phase boundary.*

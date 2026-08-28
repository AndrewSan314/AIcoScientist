# Versioned Experiment Registry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the SQLite snapshot store to an append-only, idempotent experiment registry while preserving the current active-dataset DataFrame contract.

**Architecture:** Schema version 2 adds an `experiments` parent table and scopes every scientific row by `(experiment_id, sample_id)`. Ingestion derives a deterministic identity from batch ID plus the four source hashes, appends unseen experiments atomically, and marks one experiment active. A transactional v1-to-v2 migration converts the existing snapshot into the first active experiment without losing ingestion/source lineage.

**Tech Stack:** Python, sqlite3, hashlib, pandas, pytest

---

### Task 1: Specify v1-to-v2 migration

**Files:**
- Modify: `tests/test_experiment_store.py`

**Step 1: Add a v1 database fixture**

Create a real SQLite v1 database from copied CSV sources with `schema_meta=1`, one successful ingestion run, four source-file hashes, and the four snapshot tables.

**Step 2: Write the migration test**

Call the wished-for public API `initialize_database(database)` and assert:

- schema version becomes 2,
- exactly one active experiment exists,
- every scientific table retains all rows,
- ingestion and source-file lineage counts are unchanged.

**Step 3: Run RED**

Run: `python -m pytest tests/test_experiment_store.py::test_v1_migration_preserves_data_and_lineage -q`

Expected: import/attribute failure because `initialize_database` does not exist.

### Task 2: Implement schema version 2 and migration

**Files:**
- Modify: `src/utils.py`
- Modify: `src/experiment_store.py`

**Step 1: Set schema version 2**

Change `SCHEMA_VERSION` from 1 to 2.

**Step 2: Define the v2 schema**

Add `experiments(id, batch_id, operator, created_at, source_fingerprint, ingestion_run_id, is_active)` and add `experiment_id` to the composite primary key of each scientific table.

**Step 3: Implement the migration**

Rename v1 scientific tables, create v2 tables, derive the legacy experiment identity from the latest successful run's four source hashes, copy all rows, drop temporary tables, and update `schema_meta` in one transaction.

**Step 4: Add `initialize_database`**

Expose database creation/migration as a public function used by ingestion and tests.

**Step 5: Run GREEN**

Run: `python -m pytest tests/test_experiment_store.py::test_v1_migration_preserves_data_and_lineage -q`

Expected: pass.

### Task 3: Specify append-only and idempotent ingestion

**Files:**
- Modify: `tests/test_experiment_store.py`

**Step 1: Write a multi-batch history test**

Ingest the same valid files under `batch-1` and `batch-2`; assert two experiments coexist and default loading returns `batch-2` while explicit loading returns `batch-1`.

**Step 2: Write an idempotency test**

Ingest the same files twice with the same batch ID; assert only one experiment and one set of scientific rows exists, while both ingestion attempts are recorded.

**Step 3: Strengthen rollback coverage**

After a valid active experiment, attempt an invalid second batch and assert the active experiment and prior rows are unchanged.

**Step 4: Run RED**

Run: `python -m pytest tests/test_experiment_store.py -q`

Expected: failures because `ingest_csvs` has no batch/operator metadata and the schema still replaces snapshots.

### Task 4: Implement experiment history and active loading

**Files:**
- Modify: `src/experiment_store.py`

**Step 1: Derive logical experiment identity**

Compute the source fingerprint from sorted dataset hashes plus `batch_id`; derive a deterministic experiment ID. If `batch_id` is omitted, derive a stable demo batch ID from the hashes.

**Step 2: Append unseen experiments atomically**

Insert the experiment and four scoped table snapshots without deleting history. On duplicate fingerprint, reuse the experiment and do not duplicate scientific rows. Record source lineage and a successful ingestion run for every attempt.

**Step 3: Maintain the active experiment**

Mark the imported/reused experiment active and all others inactive in the same transaction.

**Step 4: Preserve the read contract**

Make `load_source_tables(database, experiment_id=None)` return active experiment data by default, omit `experiment_id` from returned DataFrames, and support explicit historical reads.

**Step 5: Run GREEN**

Run: `python -m pytest tests/test_experiment_store.py -q`

Expected: all registry tests pass.

### Task 5: Verify downstream compatibility

**Files:**
- Modify only if a failing test proves a compatibility defect.

**Step 1: Run affected flow tests**

Run: `python -m pytest tests/test_dataset.py tests/test_model.py tests/test_recommend.py -q`

Expected: pass without changing downstream APIs.

**Step 2: Run the full suite**

Run: `python -m pytest tests -q`

Expected: all tests pass.

### Task 6: Document and close Phase 2

**Files:**
- Modify: `README.md`
- Modify: `.planning/PROJECT.md`
- Modify: `.planning/REQUIREMENTS.md`
- Modify: `.planning/STATE.md`
- Modify: `docs/plans/production-roadmap.md`

**Step 1: Document schema v2 and operator semantics**

Document historical experiments, active reads, idempotency, automatic demo batch IDs, and explicit `batch_id`/`operator` metadata for real imports.

**Step 2: Run final verification**

Run:

```text
python -m pytest tests -q
python -m compileall -q src app run_pipeline.py
python -m pip check
python run_pipeline.py
```

Then verify `PRAGMA integrity_check`, `PRAGMA foreign_key_check`, schema version 2, one active experiment, and preserved historical experiments.

**Step 3: Check change scope**

Refresh GitNexus, run `detect_changes(scope="all")` when the repository has a valid `HEAD`, and otherwise record the unborn-repository limitation plus the pre-change HIGH-risk analysis.

No commit is included because the repository currently has no tracked baseline/HEAD.

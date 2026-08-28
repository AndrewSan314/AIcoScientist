# Production Data Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make SQLite the atomic, traceable source of truth for the four scientific input datasets without changing downstream model behavior.

**Architecture:** Add one standard-library SQLite module for schema, validation, ingestion, lineage, and reads. Route the existing master dataset builder through it while retaining CSV import/export and all existing feature engineering.

**Tech Stack:** Python, sqlite3, hashlib, pandas, pytest

---

### Task 1: Specify SQLite ingestion behavior with tests

**Files:**
- Create: `tests/test_experiment_store.py`

**Step 1: Write the failing tests**

Add focused tests for successful ingestion/lineage, rollback after invalid sample IDs, and loading stored tables.

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_experiment_store.py -q`

Expected: collection fails because `src.experiment_store` does not exist.

### Task 2: Implement the minimal experiment store

**Files:**
- Create: `src/experiment_store.py`
- Modify: `src/utils.py`

**Step 1: Add paths and schema constants**

Add `DATABASE_FILE` and `SCHEMA_VERSION` to the existing utility constants.

**Step 2: Implement schema and validation**

Use `sqlite3`, explicit table DDL, required-column checks, duplicate-ID checks, identical-ID-set checks, numeric/null checks, and the 100 wt% composition rule.

**Step 3: Implement transactional ingestion and lineage**

Hash each CSV with SHA-256, replace the four snapshot tables in one transaction, and record the ingestion run/source files.

**Step 4: Implement reads**

Return the four source tables as pandas DataFrames ordered by `sample_id`.

**Step 5: Run tests to verify GREEN**

Run: `python -m pytest tests/test_experiment_store.py -q`

Expected: all focused tests pass.

### Task 3: Integrate the master dataset builder

**Files:**
- Modify: `src/build_dataset.py`
- Modify: `tests/test_dataset.py`

**Step 1: Write a failing integration assertion**

Assert that running the current pipeline creates `data/experiments.db` and records a successful ingestion.

**Step 2: Run the integration test to verify RED**

Run: `python -m pytest tests/test_dataset.py -q`

Expected: failure because the builder does not yet ingest through SQLite.

**Step 3: Route the builder through SQLite**

Keep `ensure_sample_data()`, then call the experiment-store ingest/load functions instead of reading CSVs directly. Leave feature engineering and output unchanged.

**Step 4: Run the integration test to verify GREEN**

Run: `python -m pytest tests/test_dataset.py -q`

Expected: pass.

### Task 4: Verify and document the phase gate

**Files:**
- Modify: `README.md`
- Modify: `.planning/PROJECT.md`
- Modify: `.planning/REQUIREMENTS.md`

**Step 1: Document SQLite operation and Phase 1 scope**

Record the database path, atomic ingestion behavior, lineage tables, limitations, and completed production requirement.

**Step 2: Run complete verification**

Run:

```text
python -m pytest tests -q
python -m compileall -q src app run_pipeline.py
python -m pip check
python run_pipeline.py
```

Expected: every command exits 0 and the pipeline produces the existing outputs plus `data/experiments.db`.

**Step 3: Verify blast radius**

Run GitNexus `detect_changes(scope="all")` and confirm only the data-foundation flow and its downstream pipeline are affected.

No commit is included because the repository currently has no tracked baseline.

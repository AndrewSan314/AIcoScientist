# Roadmap: Battery AI Co-Scientist MVP

## Phase 1 - Data Foundation

**Status:** Complete

**Goal:** Create the raw sample data and schema documentation required for the MVP.

**Requirements:** DATA-01, DATA-03

**Success criteria:**
1. Raw CSVs exist under `data/raw/`.
2. Each raw CSV has `sample_id`.
3. At least 30 synthetic rows are available.
4. `data_dictionary.md` explains all columns.

## Phase 2 - Master Dataset

**Status:** Complete

**Goal:** Merge raw process, SEM, EDX, and electrochemical data into one model-ready dataset.

**Requirements:** DATA-02

**Success criteria:**
1. `data/processed/master_dataset.csv` is created.
2. No sample IDs are lost during merge.
3. `retention_100` exists and has no missing values.
4. Engineered ratios are present.

## Phase 3 - Baseline Model

**Status:** Complete

**Goal:** Train and persist a regression baseline plus uncertainty surrogate.

**Requirements:** MODL-01, MODL-02, MODL-03

**Success criteria:**
1. Training runs without error.
2. Model artifact is saved.
3. Metrics JSON is saved.
4. Feature importance CSV is saved.
5. Model bundle includes a Gaussian Process surrogate and scaler.

## Phase 4 - Recommendation Engine

**Status:** Complete

**Goal:** Rank candidate fabrication recipes and export top-3 recommendations.

**Requirements:** RECO-01, RECO-02, RECO-03, RECO-04

**Success criteria:**
1. `outputs/recommendations.csv` contains exactly 3 rows.
2. Candidates are within the configured search space.
3. Existing recipes are excluded.
4. Each row includes predicted retention, confidence, and reason.
5. Each row includes GP uncertainty and UCB acquisition score.

## Phase 5 - Dashboard

**Status:** Complete

**Goal:** Provide a Streamlit demo surface for presentation screenshots.

**Requirements:** UI-01, UI-02

**Success criteria:**
1. Dashboard opens with `streamlit run app/streamlit_app.py`.
2. Dataset, metrics, feature importance, and recommendations render.
3. Accept, Modify, and Reject placeholders are visible.

## Phase 6 - End-to-End Check

**Status:** Complete

**Goal:** Make the MVP runnable and testable from the project root.

**Requirements:** OPS-01, OPS-02

**Success criteria:**
1. `python run_pipeline.py` creates all expected outputs.
2. `pytest tests/` passes.

---
*Roadmap completed: 2026-07-09*

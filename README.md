# Battery AI Co-Scientist

Runnable demo for optimizing nano-silicon/few-layer Ti3C2Tx MXene anode fabrication with sodium alginate binder, now with a production-hardened SQLite data foundation.

## Pipeline

```text
raw process + SEM + EDX + electrochemistry CSVs
-> validated SQLite experiment store + source lineage
-> merged feature table
-> RandomForest baseline + Gaussian Process surrogate
-> GP/UCB top-3 recipes
-> Streamlit decision dashboard
```

The MVP uses synthetic/sample data to demonstrate pipeline logic. Model performance is not scientifically meaningful until real experimental data are provided.

## Structure

```text
data/raw/       synthetic lab CSVs and SEM demo images
data/experiments.db  versioned experiments and ingestion lineage
data/processed/ merged and image-derived feature tables
src/            dataset, model, recommendation, and imaging pipeline
app/            Streamlit dashboard
outputs/        persisted model, metrics, importance, and recommendations
screenshots/    presentation-ready dashboard captures
tests/          data, model, recommendation, and SEM checks
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```

Outputs:

- `data/experiments.db`: versioned experiments, validated measurements, ingestion runs, and SHA-256 source lineage.
- `data/processed/master_dataset.csv`: process, SEM, EDX, and electrochemical records merged by `sample_id`.
- `data/processed/sem_features_extracted.csv`: threshold-segmentation features from images when demo images are present.
- `outputs/trained_model.pkl`: RandomForest baseline, GP surrogate, scaler, feature list, and fill values.
- `outputs/model_metrics.json`: holdout MAE, RMSE, and R2 for RF and GP.
- `outputs/feature_importance.csv`: RandomForest feature importance.
- `outputs/recommendations.csv`: top three unseen recipes ranked by GP/UCB acquisition.

## Data Ingestion

Each pipeline run validates all four input CSVs before training. A successful ingest atomically appends a new experiment or reuses an existing experiment with the same batch ID and source hashes. It records the source path, row count, and SHA-256 hash. Invalid schemas, duplicate or mismatched sample IDs, missing values, invalid numeric ranges, or compositions that do not total 100 wt% stop the pipeline without changing the active experiment.

The store uses schema version `2` and these operational tables:

- `schema_meta`
- `ingestion_runs`
- `source_files`
- `experiments`
- `process_data`, `sem_features`, `edx_data`, and `electrochem_data`

`ingest_csvs(..., batch_id="lab-batch-001", operator="name")` should be used for real imports. When metadata is omitted, the demo pipeline derives a deterministic batch ID from the source hashes and uses the `system` operator. `load_source_tables()` returns the active experiment; pass `experiment_id=` to read a historical experiment.

## Microscopy Demo

Download three public NCM622 cathode SEM images and extract numeric features:

```bash
python -m src.fetch_sem_demo
python -m src.sem_features
```

The Streamlit dashboard also includes an `SEM imaging` tab:

- upload an SEM image or select a demo image,
- optionally apply contrast enhancement before segmentation,
- use `models/sam_vit_b_01ec64.pth` automatically when that SAM checkpoint is present,
- report crack area fraction, crack count, crack length density, mean crack width, and particle area fraction.

The images come from the public repository accompanying Oh et al. (2024).
Three images exercise the pipeline only and are not a scientific training set.

Source and license details are recorded in
`data/raw/sem_images/SOURCE.md`.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest tests/
```

Research evidence is summarized in `literature_notes.md`. A concise slide flow
and speaking script are in `presentation_notes.md`.

## Limits

- Synthetic data only until real lab data is supplied.
- Recommendations use a lightweight GP/UCB Bayesian Optimization MVP over a small discrete recipe grid.
- SEM/EDX values for candidate recipes are mean-filled estimates because real SEM/EDX is only available after fabrication.
- SAM runs only when a compatible local checkpoint is present; otherwise the microscopy tab uses OpenCV/scikit-image segmentation.
- SQLite is currently single-writer; multi-user access and authentication are not implemented yet.
- No LLM literature mining is enabled.

## Next Steps

- Replace synthetic rows with versioned laboratory measurements.
- Feed accepted recipes and measured SEM/EDX/electrochemistry back into model retraining.
- Calibrate uncertainty and acquisition behavior on repeated real experiments.
- Train domain segmentation only after representative labeled Si/MXene images exist.

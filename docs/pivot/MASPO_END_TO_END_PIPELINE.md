# Dataset-pluggable MASPO surrogate pipeline

`scripts/run_maspo_pipeline.py` runs one reproducible, leakage-safe experiment:

`source dataset -> adapter/validation -> recipe-group split -> train-only preprocessing -> GP or ExtraTrees -> held-out metrics -> frozen artifact -> MASPO finite-pool ranking -> revalidation queue`

The pipeline never invokes ARTISTIC. ARTISTIC remains the expensive high-fidelity teacher/validator, and a candidate is only executable after a source-backed recipe mapping, published-domain particle preflight, and explicit approval.

## Inputs and data contracts

`ProcessSurrogateSample` records run and recipe identity, evidence source, stage, controls, earlier state, modalities, fidelity/horizon, final targets, and provenance. Final targets are rejected when configured as features. `DatasetManifest` fingerprints the exact source and schema. `DatasetValidationReport` is written before fitting.

Use `config/maspo/generic_tabular_example.yaml` as a mapping template. The `group_id` must identify physical recipe/family, not execution: repetitions and seeds share one group. ARTISTIC uses recipe fingerprints through `ArtisticRunDirectoryAdapter`, which only reads immutable normalized cache manifests; incomplete or insufficiently provenanced runs are rejected.

## Run the smoke pipeline

```powershell
python scripts/run_maspo_pipeline.py --config config/maspo/synthetic.yaml --output outputs/maspo_pipeline_synthetic
```

This synthetic fixture proves plumbing only; it is not battery-performance evidence. It has isolated recipe groups and never claims ARTISTIC validation.

## Outputs

The chosen output directory contains `dataset_manifest.json`, `validation_report.json`, `split_manifest.json`, `surrogate.joblib` plus metadata, held-out `predictions.csv`, `metrics.json`, `proposals.json`, `revalidation_queue.json`, and `experiment_manifest.json`.

The frozen artifact stores its dataset/split fingerprints, input and target schemas, preprocessing fingerprint, model configuration, metrics, and artifact fingerprint. Loading can require the expected dataset fingerprint and feature schema.

Metrics include MAE, RMSE, R2 when defined, Gaussian NLL and 50/80/95% interval coverage. The report also includes one finite-pool decision metric (selected-vs-oracle simple regret) and amortized prediction latency. Uncertainty is GP posterior standard deviation or ExtraTrees ensemble spread; neither is a calibrated scientific claim without held-out calibration evidence.

## MASPO connection and validation queue

`ArtifactOptimizerBackend` implements the existing optimizer boundary and is injected into `ProcessOptimizationCoordinator`. It ranks only a supplied finite pool using the frozen artifact, then writes auditable proposals. The queue is deliberately non-executable for synthetic records. For a real ARTISTIC queue item, retain its artifact/dataset/split hashes, map to a published-domain ARTISTIC recipe, run the particle-count preflight, then request approval before launching exactly the selected validation run.

# Dataset-pluggable MASPO surrogate pipeline

`scripts/run_maspo_pipeline.py` runs one reproducible, leakage-safe experiment:

`source dataset -> adapter/validation -> recipe-group split -> train-only preprocessing -> GP or ExtraTrees -> held-out metrics -> frozen artifact -> fixed-context MASPO action ranking -> revalidation queue`

The pipeline never invokes ARTISTIC. ARTISTIC remains the expensive high-fidelity teacher/validator, and a candidate is only executable after a source-backed recipe mapping, published-domain particle preflight, and explicit approval.

## Inputs and data contracts

`ProcessSurrogateSample` records run and recipe identity, evidence source, stage, controls, earlier state, modalities, fidelity/horizon, final targets, and provenance. Final targets are rejected when configured as features. `DatasetManifest` fingerprints the exact source and schema. `DatasetValidationReport` is written before fitting.

`SurrogateInputSchema` freezes one decision stage, control features, context features, feature order, and observed fidelity vocabulary. Unknown structural controls/context/fidelity, or a different stage, fail closed. Missing known values use their observed-mask columns. The present policy is deliberately stage-specific: mixed-stage training is rejected.

Use `config/maspo/generic_tabular_example.yaml` as a mapping template. The `group_id` must identify physical recipe/family, not execution: repetitions and seeds share one group. ARTISTIC uses recipe fingerprints through `ArtisticRunDirectoryAdapter`, which only reads immutable normalized cache manifests; incomplete or insufficiently provenanced runs are rejected.

## Run the smoke pipeline

```powershell
python scripts/run_maspo_pipeline.py --config config/maspo/synthetic.yaml --output outputs/maspo_pipeline_synthetic
```

This synthetic fixture proves plumbing only; it is not battery-performance evidence. It has isolated recipe groups and never claims ARTISTIC validation.

## Outputs

The chosen output directory contains `dataset_manifest.json`, `validation_report.json`, `split_manifest.json`, `surrogate.joblib` plus metadata, held-out `predictions.csv`, `metrics.json`, `proposals.json`, `revalidation_queue.json`, and `experiment_manifest.json`.

The frozen artifact stores its dataset/split fingerprints, input and target schemas, fitted preprocessor fingerprint, fitted model-state fingerprint, model configuration, metrics, and artifact fingerprint. The state hashes use dtype/shape/contiguous numeric bytes for fitted arrays; GP includes its fitted kernel/training state and internal scaler, while ExtraTrees includes every fitted tree structure. Saving writes a non-circular SHA256 for the artifact file to the sidecar. Loading checks file bytes, metadata, and recomputed in-memory state; `verify_integrity()` also detects post-load mutation before prediction or optimization.

Metrics include MAE, RMSE, R2 when defined, Gaussian NLL and 50/80/95% interval coverage. The report also includes one finite-pool decision metric (selected-vs-oracle simple regret) and amortized prediction latency. Uncertainty is GP posterior standard deviation or ExtraTrees ensemble spread; neither is a calibrated scientific claim without held-out calibration evidence.

## MASPO connection and validation queue

`ArtifactOptimizerBackend` implements the existing optimizer boundary and is injected into `ProcessOptimizationCoordinator`. It receives a typed `SurrogateDecisionContext`, then scores a finite pool containing only legal control variables. Every candidate gets the same state, observations, modalities, stage, and fidelity; `ProcessControlProposal.controls` therefore contains controls only. Held-out ranking is reported separately as `OFFLINE_RANKING_EVALUATION`, never as a closed-loop decision.

The queue is evidence-aware: synthetic records are `SYNTHETIC_TEST_ONLY`; generic sources require an ARTISTIC recipe mapping; normalized ARTISTIC records can be `READY_FOR_EXPLICIT_REVALIDATION` but never auto-run. `report.md` states claim boundaries, held-out metrics, offline ranking, latency p50/p95/p99, and proposal count.

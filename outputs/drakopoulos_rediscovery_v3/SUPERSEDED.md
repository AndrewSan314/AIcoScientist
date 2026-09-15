# Drakopoulos Rediscovery Benchmark v3 — SUPERSEDED

> [!WARNING]
> **This benchmark version (v3) has been officially SUPERSEDED by v4.**
> Do not use v3 results for final scientific claims or publications. See `outputs/drakopoulos_rediscovery_v4/` for the authoritative benchmark artifacts.

## Rationale for Superseding

The v3 rediscovery benchmark was superseded because measured-zero D30 semantics and full-engine execution provenance required correction:

1. **Measured-Zero D30 Semantics:**
   In v3, raw D30 values recorded as `0` or `0.0` (indicating early cell discharge failure before cycle 30) were improperly treated as missing/unmeasured (`None`), distorting replicate counts and completeness classifications. In v4, measured zeros are preserved as valid numeric observations (`0.0 mAh/g`), contributing correctly to recipe mean capacities and establishing exact replicate counts (`valid_d30_replicates`, `zero_d30_replicates`, `missing_d30_replicates`).

2. **Full-Engine Execution Provenance:**
   In v3, the benchmark claimed execution through the AIcoScientist production process engine, but parts of the execution trace and audit were hardcoded or evaluated outside `ProcessOptimizationCoordinator`. In v4, all candidate evaluations are routed strictly through `DrakopoulosGraphiteAdapter` -> `BatteryProcessRun` -> `InformationHorizon` -> `ProcessSurrogate` -> `SurrogateArtifact` -> `FrozenSurrogateOptimizerBackend` -> `ProcessOptimizationCoordinator`, with fail-closed runtime verification via `EngineExecutionTrace`.

3. **Complete-Recipe Selection Horizon:**
   In v3, `InformationHorizon(ProcessStage.COATING)` was misused as a proxy for pre-manufacturing selection, which locally masked downstream controls. In v4, an explicit `DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION` is enforced, exposing all pre-planned controls across all stages while strictly firewalling post-process measurements (active mass, thickness, porosity) and cycle outcomes.

4. **Scientific Precision in High-Loading Task Naming:**
   In v3, the subset of recipes with coating gap >= 150 um was described as high loading. In v4, this is explicitly designated as `HIGHER_LOADING_MEASURED_D30_PROXY` with `published_high_loading_rediscovery_status = NOT_EVALUABLE_WITH_AVAILABLE_D30`, because published >= 25 mg targets (300 um cells) lack usable cycle 30 measurements in the source dataset.

Please refer to `outputs/drakopoulos_rediscovery_v4/` and `outputs/drakopoulos_rediscovery_v4/DRAKOPOULOS_REDISCOVERY_V4_REPORT.md` for complete results and methodology.

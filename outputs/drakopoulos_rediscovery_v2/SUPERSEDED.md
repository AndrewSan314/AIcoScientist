# Drakopoulos Rediscovery Benchmark v2 — SUPERSEDED

> **STATUS: SUPERSEDED_BY_V3_HARDENING**
>
> The benchmark results in this directory (`outputs/drakopoulos_rediscovery_v2/`) have been **superseded** by the v3 hardening pass located in `outputs/drakopoulos_rediscovery_v3/`.
>
> **The v2 rediscovery result was superseded and was not used for the final scientific claim.**

---

## Reasons for Supersession

1. **D30 Incomplete-Replicate Survivorship Bias**:
   - In v2, recipe aggregation computed the mean specific discharge capacity only across cells with positive D30 values (`val.value > 0`), while reporting total replicate count from all cells.
   - Cells that failed or lacked cycle 30 measurements were silently excluded, risking survivorship bias.
   - In v3, recipes explicitly track `total_replicates`, `valid_d30_replicates`, `zero_d30_replicates`, `missing_d30_replicates`, and `recipe_eligibility_status`. Only strictly complete recipes (`STRICT_COMPLETE_RECIPE`) are admitted into the primary benchmark, with incomplete recipes preserved in an audit table.

2. **Production Process Surrogate Engine Path**:
   - In v2, the benchmark coordinator used `BoTorchBackend` directly on tabular features, meaning `AICOSCIENTIST_PROCESS_ENGINE` was executing the same wrapper as `DIRECT_BOTORCH_BASELINE`.
   - In v3, the benchmark exercises the genuine production pipeline: `DrakopoulosGraphiteAdapter` $\to$ `BatteryProcessRun` $\to$ `InformationHorizon(ProcessStage.COATING)` $\to$ `ProcessSurrogateSample` $\to$ `TrainOnlyPreprocessor` $\to$ `ProcessSurrogate` (GP) $\to$ `SurrogateArtifact` with cryptographic fingerprinting and verifiable schema integrity.

3. **Random Analytical Baseline & Rank Semantics**:
   - In v2, the analytical hypergeometric baseline assumed all top-$k$ targets were excluded from the initial design, whereas the empirical benchmark only excluded the top-1 hidden best.
   - In v2, `hidden_best_rank` was artificially held at rank 1 even after discovery.
   - In v3, the random top-$k$ baseline is conditioned on the exact empirical initial designs (and verified against brute-force combinatorial enumeration), and `hidden_best_rank` transitions to `null` (`None`) after candidate revelation.

4. **Task Separation (Unconstrained vs. High-Loading)**:
   - In v2, the unconstrained numerical optimum ($D_{30} \approx 402.25$ mAh/g, active mass $\approx 7.70$ mg, 100 $\mu$m coating gap) was reported without explicit separation from the paper's commercial high-loading / thick electrode objective.
   - In v3, `UNCONSTRAINED_D30` and `HIGH_LOADING_D30` ($\ge 16$ mg active mass, 200 $\mu$m coating gap) are evaluated and reported as distinct, non-conflated tasks.

Refer to `outputs/drakopoulos_rediscovery_v3/DRAKOPOULOS_REDISCOVERY_V3_REPORT.md` for the official, hardened scientific findings.

# NOTICE: BENCHMARK RESULTS INVALIDATED

> [!CAUTION]
> **DO NOT USE FOR SCIENTIFIC CLAIMS, SLIDES, PUBLICATIONS, OR BENCHMARKING.**
> The results, metrics, and figures in this directory (`outputs/drakopoulos_rediscovery/`) have been **formally invalidated** following a comprehensive scientific audit.
> 
> A corrected, production-compliant implementation is available in:
> **`outputs/drakopoulos_rediscovery_v2/`**

---

## Reasons for Invalidation (P0 Scientific Blockers)

### 1. P0-1: Semantic Column Mapping Failure
The previous benchmark adapter (`src/datasets/battery_process/drakopoulos_graphite.py`) relied on hardcoded positional column indices into Excel sheets with multi-row composite headers. This caused a catastrophic column-index displacement:
- Coating speed was mapped to drying temperature, producing physically impossible speeds of **120.0 m/min** (actual: 0.1–0.5 m/min).
- Coating gap was mapped to coating speed, producing gap settings of **0.3 µm** (actual: 70–300 µm).
- Active material fraction was mapped to gap size, yielding active material percentages of **300.0%** (actual: 93–96 wt%).
- Conductive additive fraction was mapped to active material fraction, assigning ~94.5% conductive carbon black.

All candidate recipes in `recipe_table.csv` and surrogate optimization trajectories in `trajectories.json` were trained on completely scrambled physics.

### 2. P0-2: Target Metric Mismatch
The previous benchmark targeted `cell_capacity_mah` at column 15. In the Drakopoulos dataset, this column represents theoretical capacity ($Active\ Mass 	imes 0.372	ext{ mAh/mg}$), not experimental cycling discharge performance. The optimizer was merely searching for the heaviest electrode (maximum coat weight), ignoring actual electrochemical performance, transport limitations, and cycle life degradation.

The true headline optimization target of Drakopoulos et al. (*Cell Reports Physical Science* 2021) is **discharge specific capacity at cycle 30 ($D_{30}$, mAh/g)** under high active mass loading.

### 3. P0-3: Production Engine Bypass
The previous benchmark script instantiated `BoTorchBackend.propose` directly as a generic black-box function, completely bypassing the production `ProcessOptimizationCoordinator`, `ProcessSearchSpace`, `InformationHorizon`, and battery manufacturing process contracts.

---

## Corrected Implementation

The audited, corrected benchmark is located in:
- Dataset Adapter: `src/datasets/battery_process/drakopoulos_graphite.py` (semantic regex header mapping, strict physical bounds assertions, fail-closed validation)
- Benchmark Runner: `src/process/benchmarks/rediscovery.py` (routes through production `ProcessOptimizationCoordinator`)
- Re-Audit Documentation: `outputs/drakopoulos_source_reaudit/`
- Valid Benchmark Deliverables: `outputs/drakopoulos_rediscovery_v2/`

Date of Invalidation: September 14, 2026  
Audit Authority: AIcoScientist Core Scientific Hardening Team

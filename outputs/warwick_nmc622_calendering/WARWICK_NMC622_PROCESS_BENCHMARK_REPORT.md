# Warwick NMC622 Pilot-Plant Calendering Benchmark Report

## 1. Executive Summary
- **Benchmark Name**: `WARWICK_NMC622_CALENDERING_REDISCOVERY` (`DOE_CONDITION_SELECTION`)
- **Official Source**: Mendeley Data, DOI: `10.17632/wwhm2frfmy.1`
- **Chemistry**: NMC622 cathode / lithium-metal half-cell
- **Evidence Kind**: `PILOT_LINE_HISTORICAL`
- **Total Unique Conditions**: 18
- **Total Cell Replicates**: 54 (3 cells per condition)
- **Primary Target**: `rate_performance_5c_over_0_2c` (MAXIMIZE)
- **Source-Observed Best Condition**: `EXP_03` (``) with mean = **0.7947 ± 0.0194**
- **Sequential Protocol**: Initial design = 3, Additional budget = 5, Seeds = 10 fixed predefined seeds (`[11, 23, 42, 67, 101, 137, 179, 223, 281, 353]`).

---

## 2. Quantitative Policy Comparison

| Metric | AIcoScientist Full Engine | Direct BoTorch Baseline | Random Baseline (Empirical) | Exact Analytical Random |
| :--- | :---: | :---: | :---: | :---: |
| **Hit@1** | 0.0% | 10.0% | 0.0% | 6.7% |
| **Hit@3** | 50.0% | 80.0% | 0.0% | 20.0% |
| **Hit@5** | 100.0% | 90.0% | 30.0% | 33.3% |
| **Top-3 Hit@5** | 100.0% | 90.0% | 70.0% | N/A* |
| **Mean Steps to Best** | 3.60 | 2.44 | 4.67 | 8.00 |
| *(Note)* | | | | *Exact top-3 analytical random baseline omitted because initial designs may already contain non-best top-3 candidates.* |
| **Simple Regret @ B=1** | 0.0532 | 0.0516 | 0.0193 | N/A |
| **Simple Regret @ B=3** | 0.0064 | 0.0326 | 0.0157 | N/A |
| **Simple Regret @ B=5** | 0.0000 | 0.0029 | 0.0092 | N/A |
| **Cumulative Regret @ B=5** | 0.0763 | 0.1266 | 0.0764 | N/A |
| **Regret AUC** | 0.0497 | 0.0993 | 0.0622 | N/A |
| **Eps-Optimal (1%)** | 100.0% | 90.0% | 70.0% | N/A |
| **Eps-Optimal (5%)** | 100.0% | 100.0% | 100.0% | N/A |

---

## 3. Scientific Invariants & Verification
1. **Firewall Integrity**: `BlindExperimentalOracle` firewall enforced; unselected candidates had targets masked.
2. **Replicate Grouping**: All 3 cell replicates for each candidate condition revealed simultaneously upon selection.
3. **No Lookahead**: Initial designs strictly excluded `EXP_03`; exact same initial recipes evaluated across all policies for each seed.
4. **Analytical Random Baseline**: Exact formula $P(B) = B / (18 - 3) = B / 15$ computed analytically without Monte Carlo noise.
5. **Full Engine Execution Path**: Process runs routed through `WarwickNMC622CalenderingAdapter` -> `BatteryProcessRun` -> `InformationHorizon(PRE_MANUFACTURING_RECIPE_SELECTION)` -> `ProcessSurrogateSample` -> `TrainOnlyPreprocessor` -> `ProcessSurrogate (GP)` -> `SurrogateArtifact` -> `ProcessOptimizationCoordinator`.

---

## 4. Generated Publication Figures
- `outputs/warwick_nmc622_calendering/figures/nmc622_doe_space.png`
- `outputs/warwick_nmc622_calendering/figures/nmc622_target_by_recipe.png`
- `outputs/warwick_nmc622_calendering/figures/nmc622_hit_rate_at_budget.png`
- `outputs/warwick_nmc622_calendering/figures/nmc622_best_so_far_vs_selection.png`
- `outputs/warwick_nmc622_calendering/figures/nmc622_simple_regret.png`
- `outputs/warwick_nmc622_calendering/figures/nmc622_engine_vs_baselines.png`

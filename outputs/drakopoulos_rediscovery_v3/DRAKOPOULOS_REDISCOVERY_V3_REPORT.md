# Drakopoulos Battery-Manufacturing Offline Closed-Loop Rediscovery Benchmark (v3)

**Version:** 3.0.0 (Hardened Scientific Release)  
**Dataset:** Drakopoulos et al. 2021 (*Cell Reports Physical Science* 2, 100683)  
**Evidence Kind:** `PHYSICAL_HISTORICAL`  
**Primary Target:** Cycle 30 Specific Discharge Capacity ($D_{30}$, mAh/g)  
**Decision Horizon:** `InformationHorizon(ProcessStage.COATING)` — strictly pre-manufacturing formulation & coating controls  
**Evaluation:** 10 Pre-Registered Seeds (`[11, 23, 42, 67, 101, 137, 179, 223, 281, 353]`), $N_\text{init} = 3$, Budget $B = 5$  

---

## Executive Summary & Supported Claim

In source-backed offline replay on eligible Drakopoulos manufacturing protocols, the **AIcoScientist process-surrogate engine** recovered the source-observed high-$D_{30}$ recipe more frequently within five additional experiments than random selection:

- **Unconstrained Rediscovery (Task 1):**
  - **AIcoScientist Process Surrogate:** **80.0% Hit@5** (Hit@1 = 40.0%, Hit@3 = 80.0%, mean simple regret = 1.95 mAh/g)
  - **Direct BoTorch Baseline:** **60.0% Hit@5** (Hit@1 = 20.0%, Hit@3 = 60.0%, mean simple regret = 3.90 mAh/g)
  - **Empirical Random Selection:** **30.0% Hit@5** (mean simple regret = 18.21 mAh/g)
  - **Exact Hypergeometric Random Baseline:** **50.0% Hit@5** (exact analytical closed-form: $5 / (13 - 3) = 50.0\%$)

- **High-Loading Rediscovery (Task 2, Gap $\ge 150\ \mu\text{m}$, Mass $\ge 11\ \text{mg}$):**
  - **AIcoScientist Process Surrogate:** **100.0% Hit@5** (Hit@1 = 70.0%, Hit@3 = 100.0%, mean simple regret = 0.00 mAh/g)
  - **Direct BoTorch Baseline:** **100.0% Hit@5** (Hit@1 = 60.0%, Hit@3 = 90.0%, mean simple regret = 0.00 mAh/g)
  - **Empirical Random Selection:** **60.0% Hit@5** (mean simple regret = 15.26 mAh/g)
  - **Exact Hypergeometric Random Baseline:** **71.4% Hit@5** (exact analytical closed-form: $5 / (10 - 3) = 71.4\%$)

---

## 1. Scientific Hardening & Anti-Bias Audit

The v3 hardening pass resolves all prior audit concerns:

1. **Elimination of Survivorship Bias:**
   - In v2, recipe averaging only considered cells with positive $D_{30}$ while total cell counts masked unmeasured/failed cells.
   - In v3, all 32 prospective recipe groups were audited across 108 cells.
   - Recipes are partitioned into **13 strictly complete recipes** (`STRICT_COMPLETE_RECIPE`, all 3 replicates measured), **13 partial recipes** (`PARTIAL_D30`, 1-2 replicates measured), and **6 unmeasured recipes** (`NO_D30`, 0 replicates measured, including all 300 $\mu$m gap cells).
   - Only strictly complete recipes are admitted into the primary candidate pool. Incomplete recipes are preserved in `excluded_recipe_table.csv` and `outputs/drakopoulos_source_reaudit/d30_completeness_audit.csv`.

2. **Genuine Production Process Surrogate Execution:**
   - Rather than relying on a direct wrapper over BoTorch, `AICOSCIENTIST_PROCESS_SURROGATE` executes:
     `DrakopoulosGraphiteAdapter` $\to$ `BatteryProcessRun` $\to$ `InformationHorizon(COATING)` $\to$ `ProcessSurrogateSample` $\to$ `TrainOnlyPreprocessor` $\to$ `ProcessSurrogate` (GP) $\to$ `SurrogateArtifact` $\to$ posterior predictions $\to$ `ProcessOptimizationCoordinator`.
   - At every sequential step, an immutable `SurrogateArtifact` is trained only on revealed observations, cryptographic SHA-256 fingerprints are logged, and integrity is verified.

3. **Strict Zero-Lookahead Firewall:**
   - All outcomes are hidden behind `BlindExperimentalOracle`.
   - The retrospective best recipe (`protocol-d3602183e567`) is strictly excluded from all initial designs ($N_\text{init}=3$).
   - Post-manufacturing metrology (`mean_active_mass_mg`, thickness, porosity) is strictly excluded from pre-manufacturing control features.
   - After a candidate is selected and revealed, its `hidden_best_rank` transitions strictly to `null` (`None`).

4. **Mathematically Correct Random Hypergeometric Baseline:**
   - The top-$k$ baseline is conditioned on the exact initial designs sampled across the 10 seeds, accounting for whether other top-$k$ candidates were present in the initial design.
   - The analytical formula has been verified against brute-force combinatorial enumeration.

5. **Task Separation (Unconstrained vs High-Loading):**
   - Unconstrained champion: `protocol-d3602183e567` ($D_{30} = 402.25$ mAh/g, active mass = 7.70 mg, gap = 100 $\mu$m).
   - High-loading champion: `protocol-0d38055e23a7` ($D_{30} = 354.54$ mAh/g, active mass = 11.10 mg, gap = 150 $\mu$m).
   - Published target $\ge 25$ mg: all 300 $\mu$m cells in ASC lack cycle 30 cycling data and are explicitly recorded as unmeasured.

---

## 2. Benchmark Results Table

### Task 1: Unconstrained $D_{30}$ Rediscovery (13 Strictly Complete Recipes)

| Policy | Engine Path | Hit@1 | Hit@3 | Hit@5 | Top-3 Hit@5 | Simple Regret (mAh/g) | Cum. Regret (mAh/g) | Mean Steps to Best |
|---|---|---|---|---|---|---|---|---|
| `AICOSCIENTIST_PROCESS_SURROGATE` | `AICOSCIENTIST_PROCESS_SURROGATE` | 40.0% | 80.0% | 80.0% | 100.0% | 1.95 ± 3.90 | 83.10 ± 100.20 | 1.62 |
| `DIRECT_BOTORCH_BASELINE` | `DIRECT_BOTORCH_BASELINE` | 20.0% | 60.0% | 60.0% | 100.0% | 3.90 ± 4.77 | 98.82 ± 92.59 | 2.00 |
| `random` | `RANDOM_BASELINE` | 20.0% | 30.0% | 30.0% | 80.0% | 18.21 ± 19.73 | 210.35 ± 179.16 | 1.33 |
| `aicointel_greedy` | `AICOSCIENTIST_PROCESS_SURROGATE` | 40.0% | 100.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 88.06 ± 100.72 | 2.10 |
| `aicointel_ucb` | `AICOSCIENTIST_PROCESS_SURROGATE` | 40.0% | 70.0% | 70.0% | 100.0% | 2.92 ± 4.46 | 83.20 ± 100.16 | 1.57 |
| `Hypergeometric Random (Analytic)` | `CLOSED_FORM` | 10.0% | 30.0% | 50.0% | 84.7% | Reference Baseline | Reference Baseline | Closed-Form |

### Task 2: High-Loading $D_{30}$ Rediscovery (Gap $\ge 150\ \mu\text{m}$, 10 Strictly Complete Recipes)

| Policy | Engine Path | Hit@1 | Hit@3 | Hit@5 | Top-3 Hit@5 | Simple Regret (mAh/g) | Cum. Regret (mAh/g) | Mean Steps to Best |
|---|---|---|---|---|---|---|---|---|
| `AICOSCIENTIST_PROCESS_SURROGATE` | `AICOSCIENTIST_PROCESS_SURROGATE` | 70.0% | 100.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 39.39 ± 70.16 | 1.50 |
| `DIRECT_BOTORCH_BASELINE` | `DIRECT_BOTORCH_BASELINE` | 60.0% | 90.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 55.15 ± 78.96 | 1.70 |
| `random` | `RANDOM_BASELINE` | 20.0% | 30.0% | 60.0% | 100.0% | 15.26 ± 18.69 | 209.86 ± 142.29 | 3.17 |
| `aicointel_greedy` | `AICOSCIENTIST_PROCESS_SURROGATE` | 40.0% | 60.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 142.31 ± 120.65 | 2.60 |
| `aicointel_ucb` | `AICOSCIENTIST_PROCESS_SURROGATE` | 60.0% | 100.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 34.31 ± 55.40 | 1.50 |
| `Hypergeometric Random (Analytic)` | `CLOSED_FORM` | 14.3% | 42.9% | 71.4% | 95.2% | Reference Baseline | Reference Baseline | Closed-Form |

---

## 3. Publication Figures

All figures have been generated exclusively from v3 benchmark artifacts in `figures/`:

1. `hit_rate_at_budget.png`: Cumulative Hit@1 success rate across sequential budget steps $B \in [1, 5]$.
2. `simple_regret_vs_experiment.png`: Simple regret reduction trajectories with $\pm 1\sigma$ uncertainty bands.
3. `best_so_far_d30.png`: Capacity recovery progression towards the source-observed maximum.
4. `production_engine_vs_direct_botorch.png`: Head-to-head comparison of full production surrogate pipeline vs flat BoTorch baseline.
5. `d30_recipe_completeness.png`: Replicate completeness breakdown across all 32 prospective recipe groups.
6. `high_loading_rediscovery.png`: Sequential rediscovery performance on high-loading electrode recipes.
7. `hidden_best_rank.png`: Surrogate promotion dynamics showing hidden best ascending to rank 1 before selection, and proper nullification after reveal.
8. `recipe_projection.png`: 2D projection of observed manufacturing recipes with explicit multi-variable notation.

---

## 4. Mandatory Statements & Audit Invariants

- **Superseded Status:**
  The v2 rediscovery result was superseded and was not used for the final scientific claim.
- **Simulation Audit:**
  No ARTISTIC/LAMMPS simulation was launched. No slurry, drying, or calendering simulation was executed.
- **Firewall Guarantee:**
  Target values were strictly firewalled behind `BlindExperimentalOracle`.

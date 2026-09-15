# Drakopoulos Battery-Manufacturing Offline Closed-Loop Rediscovery Benchmark (v4)

**Version:** 4.0.0 (Hardened Scientific Provenance Release)  
**Dataset:** Drakopoulos et al. 2021 (*Cell Reports Physical Science* 2, 100683)  
**Evidence Kind:** `PHYSICAL_HISTORICAL`  
**Primary Target:** Cycle 30 Specific Discharge Capacity ($D_{30}$, mAh/g)  
**Decision Horizon:** `DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION` — strictly pre-manufacturing formulation, coating, drying & calendering controls without lookahead into post-process metrology or cycle outcomes  
**Evaluation:** 10 Pre-Registered Seeds (`[11, 23, 42, 67, 101, 137, 179, 223, 281, 353]`), $N_\text{init} = 3$, Budget $B = 5$  

---

## Executive Summary & Supported Claim

In source-backed offline replay on strictly complete Drakopoulos manufacturing recipes, the AIcoScientist full process engine recovered the source-observed highest-D30 recipe more often within five additional experiments than expected under random selection.

- **Unconstrained Rediscovery (Task 1, 13 Strictly Complete Recipes):**
  - **AIcoScientist Full Process Engine:** **100.0% Hit@5** (Hit@1 = 30.0%, Hit@3 = 50.0%, mean simple regret = 0.00 mAh/g)
  - **Direct BoTorch Baseline:** **30.0% Hit@5** (Hit@1 = 20.0%, Hit@3 = 30.0%, mean simple regret = 10.61 mAh/g)
  - **Empirical Random Selection:** **30.0% Hit@5** (mean simple regret = 14.41 mAh/g)
  - **Exact Hypergeometric Random Baseline:** **55.6% Hit@5** (exact analytical closed-form: $5 / (12 - 3) = 55.6\%$)

- **Higher-Loading Measured-$D_{30}$ Proxy Subset (Task 2, Gap $\ge 150\ \mu\text{m}$, 10 Strictly Complete Recipes):**
  - **AIcoScientist Full Process Engine:** **100.0% Hit@5** (Hit@1 = 0.0%, Hit@3 = 90.0%, mean simple regret = 0.00 mAh/g)
  - **Direct BoTorch Baseline:** **100.0% Hit@5** (Hit@1 = 20.0%, Hit@3 = 90.0%, mean simple regret = 0.00 mAh/g)
  - **Empirical Random Selection:** **60.0% Hit@5** (mean simple regret = 15.26 mAh/g)
  - **Exact Hypergeometric Random Baseline:** **83.3% Hit@5** (exact analytical closed-form: $5 / (9 - 3) = 83.3\%$)
  - **Published High-Loading Objective:** `published_high_loading_rediscovery_status = NOT_EVALUABLE_WITH_AVAILABLE_D30` (300 $\mu$m cells reaching published $\ge 25$ mg lack usable cycle 30 cycling measurements).

---

## 1. Scientific Hardening & Anti-Bias Audit

The v4 hardening pass provides complete end-to-end scientific and provenance verification:

1. **Measured-Zero $D_{30}$ Preservation:**
   - In previous iterations, raw $D_{30} = 0$ (early cell failure before cycle 30) was incorrectly coerced to `None`. In v4, measured zero is retained as a valid numeric observation (`0.0 mAh/g`) contributing to recipe mean capacity and replicate completeness counts.
   - Auditing all 32 prospective recipe groups across 108 ASC cells partitions them into **13 strictly complete recipes** (`STRICT_COMPLETE_RECIPE`, all 3 replicates measured), **13 partial recipes** (`PARTIAL_D30`), and **6 unmeasured recipes** (`NO_D30`, including all 300 $\mu$m gap cells).

2. **Genuine Full Process Engine Execution:**
   - The benchmark routes strictly through: `DrakopoulosGraphiteAdapter` $\to$ `BatteryProcessRun` $\to$ `InformationHorizon` (`PRE_MANUFACTURING_RECIPE_SELECTION`) $\to$ `ProcessSurrogateSample` $\to$ `TrainOnlyPreprocessor` $\to$ `ProcessSurrogate` (GP) $\to$ `SurrogateArtifact` $\to$ `FrozenSurrogateOptimizerBackend` $\to$ `ProcessOptimizationCoordinator`.
   - At every sequential step, an immutable `SurrogateArtifact` is trained only on revealed observations, cryptographic SHA-256 fingerprints are logged, and integrity is verified.
   - Proposals are generated strictly through `ProcessOptimizationCoordinator.propose_recipes(...)`.

3. **Strict Zero-Lookahead Firewall:**
   - All outcomes are hidden behind `BlindExperimentalOracle`.
   - The retrospective best recipe (`protocol-c1c280b7366f`) is strictly excluded from all initial designs ($N_\text{init}=3$).
   - Post-manufacturing metrology (`mean_active_mass_mg`, thickness, porosity) is strictly excluded from pre-manufacturing control features.
   - After a candidate is selected and revealed, its `hidden_best_rank` transitions strictly to `null` (`None`).
   - Adversarial perturbation tests verify that changing unrevealed target $D_{30}$ or post-process metrology cannot alter the subsequent proposal.

4. **Mathematically Correct Random Hypergeometric Baseline:**
   - The top-$k$ baseline is conditioned on the exact initial designs sampled across the 10 seeds, accounting for whether other top-$k$ candidates were present in the initial design.
   - The analytical formula has been verified against brute-force combinatorial enumeration.

5. **Task Separation (Unconstrained vs Higher-Loading Proxy):**
   - Unconstrained champion: `protocol-c1c280b7366f` ($D_{30} = 402.25$ mAh/g, active mass = 7.70 mg, gap = 100 $\mu$m).
   - Higher-loading proxy champion: `protocol-0d38055e23a7` ($D_{30} = 354.54$ mAh/g, active mass = 11.10 mg, gap = 150 $\mu$m).
   - Published target $\ge 25$ mg: all 300 $\mu$m cells in ASC lack cycle 30 cycling data and are explicitly recorded as `NOT_EVALUABLE_WITH_AVAILABLE_D30`.

---

## 2. Benchmark Results Table

### Task 1: Unconstrained $D_{30}$ Rediscovery (13 Strictly Complete Recipes)

| Policy | Engine Path | Hit@1 | Hit@3 | Hit@5 | Top-3 Hit@5 | Simple Regret (mAh/g) | Cum. Regret (mAh/g) | Mean Steps to Best |
|---|---|---|---|---|---|---|---|---|
| `AICOSCIENTIST_PROCESS_SURROGATE` | `AICOSCIENTIST_PROCESS_SURROGATE` | 30.0% | 50.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 97.33 ± 69.96 | 2.80 |
| `DIRECT_BOTORCH_BASELINE` | `DIRECT_BOTORCH_BASELINE` | 20.0% | 30.0% | 30.0% | 100.0% | 10.61 ± 13.11 | 142.72 ± 90.21 | 1.33 |
| `random` | `RANDOM_BASELINE` | 20.0% | 30.0% | 30.0% | 80.0% | 14.41 ± 17.18 | 178.64 ± 141.70 | 1.33 |
| `aicointel_greedy` | `AICOSCIENTIST_PROCESS_SURROGATE` | 30.0% | 50.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 97.33 ± 69.96 | 2.80 |
| `aicointel_ucb` | `AICOSCIENTIST_PROCESS_SURROGATE` | 30.0% | 50.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 97.33 ± 69.96 | 2.80 |
| `Hypergeometric Random (Analytic)` | `CLOSED_FORM` | 11.1% | 33.3% | 55.6% | 89.3% | Reference Baseline | Reference Baseline | Closed-Form |

### Task 2: Higher-Loading Measured-$D_{30}$ Proxy Subset Rediscovery (Gap $\ge 150\ \mu\text{m}$, 10 Strictly Complete Recipes)

| Policy | Engine Path | Hit@1 | Hit@3 | Hit@5 | Top-3 Hit@5 | Simple Regret (mAh/g) | Cum. Regret (mAh/g) | Mean Steps to Best |
|---|---|---|---|---|---|---|---|---|
| `AICOSCIENTIST_PROCESS_SURROGATE` | `AICOSCIENTIST_PROCESS_SURROGATE` | 0.0% | 90.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 90.75 ± 84.95 | 2.50 |
| `DIRECT_BOTORCH_BASELINE` | `DIRECT_BOTORCH_BASELINE` | 20.0% | 90.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 84.39 ± 81.61 | 2.20 |
| `random` | `RANDOM_BASELINE` | 20.0% | 30.0% | 60.0% | 100.0% | 15.26 ± 18.69 | 177.71 ± 111.50 | 3.17 |
| `aicointel_greedy` | `AICOSCIENTIST_PROCESS_SURROGATE` | 0.0% | 90.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 90.75 ± 84.95 | 2.50 |
| `aicointel_ucb` | `AICOSCIENTIST_PROCESS_SURROGATE` | 20.0% | 90.0% | 100.0% | 100.0% | 0.00 ± 0.00 | 83.12 ± 90.84 | 2.30 |
| `Hypergeometric Random (Analytic)` | `CLOSED_FORM` | 16.7% | 50.0% | 83.3% | 100.0% | Reference Baseline | Reference Baseline | Closed-Form |

---

## 3. Publication Figures

All figures have been generated exclusively from v4 benchmark artifacts in `figures/`:

1. `hit_rate_at_budget.png`: Cumulative Hit@1 success rate across sequential budget steps $B \in [1, 5]$.
2. `simple_regret_vs_experiment.png`: Simple regret reduction trajectories with $\pm 1\sigma$ uncertainty bands.
3. `best_so_far_d30.png`: Capacity recovery progression towards the source-observed maximum.
4. `full_engine_vs_direct_botorch.png`: Head-to-head comparison of full production surrogate pipeline vs flat BoTorch baseline.
5. `d30_recipe_completeness.png`: Replicate completeness breakdown across all 32 prospective recipe groups.
6. `higher_loading_proxy_rediscovery.png`: Sequential rediscovery performance on higher-loading measured-D30 proxy subset.
7. `hidden_best_rank.png`: Surrogate promotion dynamics showing hidden best ascending to rank 1 before selection, and proper nullification after reveal.
8. `recipe_projection.png`: 2D projection of observed manufacturing recipes with explicit multi-variable notation.

---

## 4. Mandatory Statements & Audit Invariants

- **Superseded Status:**
  The v3 rediscovery benchmark was superseded because measured-zero D30 semantics and full-engine execution provenance required correction.
- **Simulation Audit:**
  No ARTISTIC/LAMMPS simulation was launched. No slurry, drying, or calendering simulation was executed.
- **Firewall Guarantee:**
  Target values were strictly firewalled behind `BlindExperimentalOracle`.
- **Higher-Loading Proxy Semantics:**
  published_high_loading_rediscovery_status = NOT_EVALUABLE_WITH_AVAILABLE_D30

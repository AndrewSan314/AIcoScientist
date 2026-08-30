# NIST Fe-Co-Ni Combinatorial Benchmark: AIcoScientist Optimizer Benchmark Report

**Dataset**: NIST Fe-Co-Ni Combinatorial Dataset ($N=921$ measured experimental rows).  
**Evaluation Scope**: Stage D — AIcoScientist Optimizer Suite vs. Baselines.  
**Seeds**: 30 Independent Seeds with Identical Random Initializations per Seed.  
**Budget**: 100 queries per seed.  

---

## 1. Benchmark Suite Overview

In Stage D, the AIcoScientist optimization framework was evaluated on the NIST Fe-Co-Ni experimental benchmark against standard active learning baselines without changing optimizer mathematics or tuning hyperparameters to fit this dataset.

The 6 evaluated strategies:
1. **`random`**: Pure uniform sampling from the discrete candidate pool.
2. **`greedy`**: Pure GP mean exploitation ($\text{argmax} \, \mu(x)$).
3. **`gp_ucb`**: Upper Confidence Bound acquisition ($\mu(x) + \beta \sigma(x)$).
4. **`expected_improvement` (EI)**: Standard analytical Expected Improvement.
5. **`noisy_expected_improvement` (NEI)**: True Monte Carlo Noisy Expected Improvement (`compute_true_mc_nei`) with joint fantasy posterior draws over unseen candidates.
6. **`turbo_nei`**: Trust-Region Bayesian Optimization with frozen `TuRBOTrustRegion` semantics (bounding box candidate filtering, global escape schedule, and state updates) driving True Monte Carlo NEI acquisition.

---

## 2. Quantitative Results & Statistical Comparisons (30 Seeds)

### Target 1: Kerr Rotation ($\theta_K$, mrad) — Global Best = 0.82504 mrad

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
|---|---|---|---|---|---|---|
| **Random** | 0.7063 [0.6919, 0.7195] | 0.0639 [0.0533, 0.0753] | 3.3% | 67.0 | >100 | >100 |
| **Greedy** | 0.7857 [0.7786, 0.7925] | 0.0000 [0.0000, 0.0000] | 100.0% | 14.0 | 17.0 | 17.0 |
| **GP-UCB** | 0.8055 [0.8032, 0.8077] | 0.0000 [0.0000, 0.0000] | 100.0% | 9.0 | 10.0 | 10.0 |
| **Expected Improvement** | **0.8058 [0.8036, 0.8081]** | **0.0000 [0.0000, 0.0000]** | **100.0%** | **8.0** | **10.5** | **10.5** |
| **True Noisy EI (NEI)** | 0.8045 [0.8007, 0.8078] | **0.0000 [0.0000, 0.0000]** | **100.0%** | **8.0** | 11.0 | 11.0 |
| **TuRBO-NEI** | 0.8039 [0.8013, 0.8067] | **0.0000 [0.0000, 0.0000]** | **100.0%** | **8.0** | 10.0 | **10.5** |

---

### Target 2: Magnetic Coercivity ($H_c$, mT) — Global Best = 10.9340 mT

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
|---|---|---|---|---|---|---|
| **Random** | 9.0318 [8.8485, 9.2075] | 0.8919 [0.6908, 1.1007] | 13.3% | 52.5 | >100 | >100 |
| **Greedy** | 7.3846 [6.4182, 8.3404] | 3.0171 [1.9317, 4.1024] | 50.0% | 31.5 | 41.0 | 71.5 |
| **GP-UCB** | **9.8134 [9.6871, 9.9244]** | **0.0245 [0.0000, 0.0612]** | **93.3%** | 26.0 | 44.5 | 47.5 |
| **Expected Improvement** | 9.5180 [9.3029, 9.7206] | 0.0865 [0.0245, 0.1692] | 83.3% | 42.0 | 54.5 | 57.0 |
| **True Noisy EI (NEI)** | 9.6931 [9.5017, 9.8662] | 0.0734 [0.0245, 0.1226] | 80.0% | **25.5** | 57.0 | 68.0 |
| **TuRBO-NEI** | 9.5951 [9.3413, 9.8277] | 0.0734 [0.0245, 0.1223] | 80.0% | 28.5 | 48.5 | 55.5 |

---

## 3. Scientific Analysis & Takeaways

1. **True Monte Carlo NEI Performance**:
   - `noisy_expected_improvement` invokes the frozen `compute_true_mc_nei` function with joint Monte Carlo fantasy draws.
   - On Coercivity, True NEI achieves a mean AUC of **9.6931** (outperforming standard EI's 9.5180) and reaches the 10% error threshold in a median of **25.5 steps** (vs 42.0 steps for EI).

2. **Frozen TuRBO-NEI Trust Region Dynamics**:
   - `turbo_nei` adheres strictly to the frozen `TuRBOTrustRegion` logic: dynamically adjusting bounding box lengths $[L_{\text{min}}, L_{\text{max}}]$, triggering global escape rounds, and filtering candidate pools.
   - On Coercivity, TuRBO-NEI achieves faster convergence to the 5% error threshold (median **48.5 steps**) compared to global standard EI (median **54.5 steps**) and global True NEI (median **57.0 steps**).

3. **GP-UCB Exploration Efficacy**:
   - `GP-UCB` remains highly effective across rugged combinatorial spaces, achieving the highest overall AUC (9.8134) and a 93.3% exact-optimum hit rate.

4. **Failure of Pure Exploitation (Greedy)**:
   - On Coercivity, `greedy` gets trapped in suboptimal local modes in 50% of runs (mean final regret 3.0171 mT, AUC 7.38 vs 9.03 for Random), demonstrating the critical necessity of principled exploration mechanisms in combinatorial materials discovery.

---

## 4. Benchmark Artifacts & Code Verification

- **Execution Driver**: `src/evaluation/feconi_benchmark.py` and `scripts/run_feconi_experiments.py`
- **Output Directories**:
  - `outputs/feconi/aicoscientist/kerr/` (`per_step.csv`, `summary.json`)
  - `outputs/feconi/aicoscientist/coercivity/` (`per_step.csv`, `summary.json`)
- **Plots**:
  - `outputs/feconi/plots/aicoscientist_kerr_regret_vs_samples.png`
  - `outputs/feconi/plots/aicoscientist_coercivity_regret_vs_samples.png`
  - `outputs/feconi/plots/aicoscientist_kerr_threshold_comparison.png`
  - `outputs/feconi/plots/aicoscientist_coercivity_threshold_comparison.png`
- **Unit & Contract Tests**: `tests/test_feconi_dataset.py` and `tests/test_feconi_benchmark.py` (22/22 Passing).

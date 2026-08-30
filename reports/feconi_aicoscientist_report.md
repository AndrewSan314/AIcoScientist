# NIST Fe-Co-Ni Combinatorial Benchmark: AIcoScientist Optimizer Benchmark Report

**Dataset**: NIST Fe-Co-Ni Combinatorial Dataset ($N=921$ measured experimental rows, Wang et al., *Benchmarking Active Learning Strategies for Materials Optimization and Discovery*, 2022).  
**Evaluation Scope**: Stage D — AIcoScientist Optimizer Suite vs. Baselines.  
**Seeds**: 30 Independent Seeds with Identical Random Initializations per Seed.  
**Budget**: 100 queries per seed.  

---

## 1. Benchmark Suite Overview

In Stage D, the AIcoScientist optimization framework was evaluated on the NIST Fe-Co-Ni experimental benchmark against standard active learning baselines, strictly reusing the GP surrogate configuration and TuRBO trust region observation lifecycle of the frozen `ClosedLoopOptimizer`:

The 6 evaluated strategies:
1. **`random`**: Pure uniform sampling from the discrete candidate pool.
2. **`greedy`**: Pure GP mean exploitation ($\text{argmax} \, \mu(x)$).
3. **`gp_ucb`**: Upper Confidence Bound acquisition ($\mu(x) + \beta \sigma(x)$).
4. **`expected_improvement` (EI)**: Standard analytical Expected Improvement.
5. **`noisy_expected_improvement` (NEI)**: True Monte Carlo Noisy Expected Improvement (`compute_true_mc_nei`) with joint fantasy posterior draws over unseen candidates.
6. **`turbo_nei`**: Trust-Region Bayesian Optimization with frozen `TuRBOTrustRegion` semantics:
   - Default frozen parameters (`init_length=0.8`, `min_length=0.05`, `max_length=1.6`, `success_tolerance=3`, `failure_tolerance=5`, `success_delta=1.0`, `global_escape_frequency=6`).
   - Finite pool candidate bounding box filtering.
   - Post-observation lifecycle matching `ClosedLoopOptimizer`: observation added to $D_{t+1}$, GP refitted on $D_{t+1}$, full latent posterior cross-covariance $(\sigma^2_{\text{cand}}, \sigma^2_{\text{inc}}, \text{Cov}(\text{cand}, \text{inc}))$ computed and passed to `TuRBOTrustRegion.update(...)`.

---

## 2. Quantitative Results & Statistical Comparisons (30 Seeds)

### Target 1: Kerr Rotation ($\theta_K$, mrad) — Global Best = 0.82504 mrad

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
|---|---|---|---|---|---|---|
| **Random** | 0.7063 [0.6919, 0.7195] | 0.0639 [0.0533, 0.0753] | 3.3% | 67.0 | >100 | >100 |
| **Greedy** | 0.7651 [0.7482, 0.7807] | 0.0000 [0.0000, 0.0000] | 100.0% | 15.5 | 17.5 | 17.5 |
| **GP-UCB** | 0.8037 [0.8010, 0.8062] | 0.0000 [0.0000, 0.0000] | 100.0% | 9.0 | 10.0 | 10.0 |
| **Expected Improvement** | 0.8028 [0.7997, 0.8055] | 0.0000 [0.0000, 0.0000] | 100.0% | 9.0 | 12.0 | 12.5 |
| **True Noisy EI (NEI)** | **0.8070 [0.8045, 0.8093]** | **0.0000 [0.0000, 0.0000]** | **100.0%** | **7.0** | **11.0** | **11.0** |
| **TuRBO-NEI** | 0.7924 [0.7804, 0.8018] | **0.0000 [0.0000, 0.0000]** | **100.0%** | 10.0 | 22.5 | 22.5 |

---

### Target 2: Magnetic Coercivity ($H_c$, mT) — Global Best = 10.9340 mT

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
|---|---|---|---|---|---|---|
| **Random** | 9.0318 [8.8485, 9.2075] | 0.8919 [0.6908, 1.1007] | 13.3% | 52.5 | >100 | >100 |
| **Greedy** | 8.0946 [7.4064, 8.7322] | 0.6791 [0.2072, 1.3314] | 53.3% | 44.5 | 90.0 | 100.0 |
| **GP-UCB** | 9.5102 [9.3514, 9.6588] | 0.3960 [0.3078, 0.4811] | 20.0% | 26.5 | 81.5 | >100 |
| **Expected Improvement** | 9.3722 [9.1447, 9.5886] | 0.3607 [0.2654, 0.4638] | 26.7% | 28.5 | 75.0 | >100 |
| **True Noisy EI (NEI)** | **9.7265 [9.5094, 9.9290]** | 0.3618 [0.2627, 0.4560] | 30.0% | 26.0 | **66.0** | >100 |
| **TuRBO-NEI** | 9.3362 [8.9848, 9.6625] | **0.2794 [0.1621, 0.3999]** | **53.3%** | **24.0** | 74.5 | **91.0** |

---

## 3. Scientific Analysis & Takeaways

1. **True Monte Carlo NEI Performance**:
   - `noisy_expected_improvement` invokes the frozen `compute_true_mc_nei` function with joint Monte Carlo fantasy draws.
   - On Coercivity, True NEI achieves the highest overall AUC (**9.7265**), outperforming standard EI's 9.3722 and GP-UCB's 9.5102, and reaches the 5% error threshold in a median of **66.0 steps** (vs 75.0 steps for EI and 81.5 steps for GP-UCB).

2. **Frozen TuRBO-NEI Trust Region Dynamics**:
   - `turbo_nei` adheres strictly to the frozen `TuRBOTrustRegion` lifecycle: refitting GP on $D_{t+1}$, computing exact candidate-incumbent covariance $\text{Cov}(\text{cand}, \text{inc})$, and updating trust region states.
   - On Coercivity, TuRBO-NEI achieves the **lowest mean final regret (0.2794 mT)** and the **highest exact optimum hit rate (53.3%)** among all Bayesian optimization methods, with the fastest median steps to 10% error (**24.0 steps**).

3. **Failure of Pure Exploitation (Greedy)**:
   - On Coercivity, `greedy` suffers significant variance and gets trapped in suboptimal local modes in nearly half of runs (mean final regret 0.6791 mT, AUC 8.09 vs 9.03 for Random), demonstrating the critical necessity of principled exploration mechanisms in combinatorial materials discovery.

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
- **Unit & Contract Tests**: `tests/test_feconi_dataset.py` and `tests/test_feconi_benchmark.py` (24/24 Passing).

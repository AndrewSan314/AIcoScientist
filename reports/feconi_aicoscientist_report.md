# NIST Fe-Co-Ni Combinatorial Benchmark: AIcoScientist Optimizer Benchmark Report

**Dataset**: NIST Fe-Co-Ni Combinatorial Dataset ($N=921$ measured experimental rows).  
**Evaluation Scope**: Stage D — AIcoScientist Optimizer Suite vs. Baselines.  
**Seeds**: 30 Independent Seeds with Identical Random Initializations per Seed.  
**Budget**: 100 queries per seed.  

---

## 1. Benchmark Suite Overview

In Stage D, the full AIcoScientist optimization framework was evaluated on the NIST Fe-Co-Ni experimental benchmark against standard active learning baselines without changing optimizer mathematics or tuning hyperparameters to fit this dataset.

The 6 evaluated strategies:
1. **`random`**: Pure uniform sampling from the discrete candidate pool.
2. **`greedy`**: Pure GP mean exploitation ($\text{argmax} \, \mu(x)$).
3. **`gp_ucb`**: Upper Confidence Bound acquisition ($\mu(x) + \beta \sigma(x)$).
4. **`expected_improvement` (EI)**: Standard analytical Expected Improvement.
5. **`noisy_expected_improvement` (NEI)**: Denoised Expected Improvement with latent GP variance estimation.
6. **`turbo_nei`**: Trust-Region Bayesian Optimization with NEI acquisition, operating on discrete pool candidates within local hyper-rectangles.

---

## 2. Quantitative Results & Statistical Comparisons (30 Seeds)

### Target 1: Kerr Rotation ($\theta_K$, mrad) — Global Best = 0.82504 mrad

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | 1% Success Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to 1% |
|---|---|---|---|---|---|---|
| **Random** | 0.7063 [0.6919, 0.7195] | 0.0639 [0.0533, 0.0753] | 3.3% | 67.0 | >100 | >100 |
| **Greedy** | 0.7857 [0.7786, 0.7925] | 0.0000 [0.0000, 0.0000] | 100.0% | 14.0 | 17.0 | 17.0 |
| **GP-UCB** | 0.8055 [0.8032, 0.8077] | 0.0000 [0.0000, 0.0000] | 100.0% | 9.0 | 10.0 | 10.0 |
| **Expected Improvement** | **0.8058 [0.8036, 0.8081]** | **0.0000 [0.0000, 0.0000]** | **100.0%** | **8.0** | **10.5** | **10.5** |
| **Noisy EI (NEI)** | **0.8058 [0.8035, 0.8081]** | **0.0000 [0.0000, 0.0000]** | **100.0%** | **8.0** | **10.5** | **10.5** |
| **TuRBO-NEI** | 0.7998 [0.7946, 0.8042] | 0.0000 [0.0000, 0.0000] | 100.0% | **7.5** | 12.5 | 12.5 |

---

### Target 2: Magnetic Coercivity ($H_c$, mT) — Global Best = 10.9340 mT

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | 1% Success Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to 1% |
|---|---|---|---|---|---|---|
| **Random** | 9.0318 [8.8485, 9.2075] | 0.8919 [0.6908, 1.1007] | 13.3% | 52.5 | >100 | >100 |
| **Greedy** | 7.3846 [6.4182, 8.3404] | 3.0171 [1.9317, 4.1024] | 50.0% | 31.5 | 41.0 | 71.5 |
| **GP-UCB** | **9.8134 [9.6871, 9.9244]** | **0.0245 [0.0000, 0.0612]** | **93.3%** | 26.0 | 44.5 | 47.5 |
| **Expected Improvement** | 9.5180 [9.3029, 9.7206] | 0.0865 [0.0245, 0.1692] | 83.3% | 42.0 | 54.5 | 57.0 |
| **Noisy EI (NEI)** | 9.5232 [9.2342, 9.7705] | **0.0245 [0.0000, 0.0612]** | **93.3%** | 29.0 | 41.5 | 45.5 |
| **TuRBO-NEI** | 9.7335 [9.3891, 10.0585] | 0.0367 [0.0000, 0.0734] | 90.0% | **26.0** | **37.5** | **43.0** |

---

## 3. Scientific Analysis & Takeaways

1. **TuRBO-NEI Efficiency on Complex Topologies**:
   - On the rugged Coercivity landscape, **`TuRBO-NEI` achieves the fastest convergence to the 5% and 1% error thresholds** (median 37.5 and 43.0 queries respectively), outperforming standard global EI (median 54.5 and 57.0 queries).
   - The trust-region mechanism focuses sampling on promising high-coercivity phase boundaries, avoiding wasteful explorations in low-coercivity Fe-rich regions.

2. **Denoised / Noisy Expected Improvement (NEI) Robustness**:
   - `NEI` consistently beats standard analytical `EI` in final regret on Coercivity (mean final regret 0.0245 vs 0.0865 mT) and improves the 1% success rate from 83.3% to **93.3%**.
   - By integrating over the GP observation noise, NEI avoids over-sampling near-identical noisy candidates.

3. **GP-UCB Exploration Power**:
   - `GP-UCB` achieves the highest overall AUC (9.8134) on Coercivity, matching the behavior observed in the paper reproduction benchmark.

4. **Failure of Pure Exploitation (Greedy)**:
   - On Coercivity, `greedy` remains trapped in local optima in 50% of runs, with a large final regret of 3.017 mT and an AUC (7.38) that is substantially worse than uniform Random sampling (9.03).
   - This validates the necessity of principled exploration-exploitation balancing in real combinatorial materials optimization.

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

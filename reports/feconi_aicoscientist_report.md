# NIST Fe-Co-Ni Combinatorial Benchmark: AIcoScientist Optimizer Benchmark Report

> [!WARNING]
> **LEGACY NATIVE OPTIMIZER RESULT**  
> Historical only. Superseded for production optimizer evaluation by BoTorchBackend migration.  
> Do not use this report as evidence of current production optimizer performance.

**Dataset**: NIST Fe-Co-Ni Combinatorial Thin-Film Dataset ($N=921$ measured experimental compositions, Wang et al., *Benchmarking Active Learning Strategies for Materials Optimization and Discovery*, 2022).  
**Evaluation Scope**: Stage D — AIcoScientist Optimizer Suite vs. Baselines (Scale-Invariant Freeze).  
**Seeds**: 30 Independent Seeds with Identical Random Initializations per Seed.  
**Budget**: 100 queries per seed (Budget = 100).  

---

## 1. Benchmark Suite Overview & Correctness Improvements

The Fe-Co-Ni benchmark evaluates closed-loop Bayesian optimization on physical combinatorial materials measurements:
1. **Single-Observation Lifecycle Fix**: Corrected a bug where observations were appended twice per query. At step $t$, the training dataset cardinality is strictly $t$, ensuring valid GP likelihood updates and accurate surrogate uncertainties.
2. **Canonical Scale Invariance**: Standardized on canonical $\xi = 0.0$ for analytic EI and True Monte Carlo NEI, and canonical $\delta_{\text{succ}} = 0.0$ for TuRBO posterior success evaluation ($P(f_{\text{cand}} > f_{\text{inc}} \mid \mathcal{D}) \ge 0.6$).

### Evaluated Strategies:
1. **`random`**: Uniform random sampling from the discrete candidate pool.
2. **`greedy`**: Pure GP mean exploitation ($\text{argmax} \, \mu(x)$).
3. **`gp_ucb`**: Upper Confidence Bound acquisition ($\mu(x) + \beta \sigma(x)$, $\beta=2.0$).
4. **`expected_improvement` (EI)**: Analytic Expected Improvement ($\xi=0.0$).
5. **`noisy_expected_improvement` (NEI)**: True Monte Carlo Noisy Expected Improvement (`compute_true_mc_nei`, $\xi=0.0$, 256 joint fantasies).
6. **`turbo_nei`**: Trust-Region Bayesian Optimization with scale-invariant `TuRBOTrustRegion` semantics (`init_length=0.8`, `min_length=0.05`, `max_length=1.6`, `success_tolerance=3`, `failure_tolerance=5`, `success_delta=0.0`, `global_escape_frequency=6`).

---

## 2. Quantitative Results & Statistical Comparisons (30 Seeds)

### Target 1: Kerr Rotation ($\theta_K$, mrad) — Global Best = 0.82504 mrad

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.7063 [0.6919, 0.7195] | 0.0639 [0.0533, 0.0753] | 3.3% | 67.0 | >100 | >100 |
| **Greedy** | 0.7651 [0.7482, 0.7807] | 0.0000 [0.0000, 0.0000] | **100.0%** | 15.5 | 17.5 | 17.5 |
| **GP-UCB** | 0.8037 [0.8010, 0.8062] | 0.0000 [0.0000, 0.0000] | **100.0%** | 9.0 | 10.0 | **11.0** |
| **Expected Improvement** | 0.8028 [0.7997, 0.8055] | 0.0000 [0.0000, 0.0000] | **100.0%** | 9.0 | 12.0 | 12.5 |
| **True Noisy EI (NEI)** | **0.8070 [0.8045, 0.8093]** | **0.0000 [0.0000, 0.0000]** | **100.0%** | **7.0** | **11.0** | **11.0** |
| **TuRBO-NEI** | 0.7924 [0.7804, 0.8018] | **0.0000 [0.0000, 0.0000]** | **100.0%** | 10.0 | 22.5 | 22.5 |

---

### Target 2: Magnetic Coercivity ($H_c$, mT) — Global Best = 10.9340 mT

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random** | 9.0318 [8.8485, 9.2075] | 0.8919 [0.6908, 1.1007] | 13.3% | 52.5 | >100 | >100 |
| **Greedy** | 8.0946 [7.4064, 8.7322] | 0.6791 [0.2072, 1.3314] | 53.3% | 44.5 | 90.0 | 100.0 |
| **GP-UCB** | 9.5102 [9.3514, 9.6588] | 0.3960 [0.3078, 0.4811] | 20.0% | 26.5 | 81.5 | >100 |
| **Expected Improvement** | 9.3722 [9.1447, 9.5886] | 0.3607 [0.2654, 0.4638] | 26.7% | 28.5 | 75.0 | >100 |
| **True Noisy EI (NEI)** | **9.7265 [9.5094, 9.9290]** | 0.3618 [0.2627, 0.4560] | 30.0% | 26.0 | **66.0** | >100 |
| **TuRBO-NEI** | 9.3362 [8.9848, 9.6625] | **0.2794 [0.1621, 0.3999]** | **53.3%** | **24.0** | 74.5 | **91.0** |

---

## 3. Scientific Analysis & Key Takeaways

1. **Exact Optimum Discovery on Kerr Rotation**:
   - Every BO policy reaches the exact global optimum ($\theta_K = 0.82504$ mrad) in 100% of seeds, whereas Random sampling succeeds in only 3.3% of seeds.
   - True Monte Carlo NEI achieves the highest trajectory AUC (**0.8070**) and reaches the 10% error threshold in just **7.0 steps** (median).
2. **Superior Local Mode Traversal on Coercivity**:
   - On the complex, multi-modal Coercivity landscape, **TuRBO-NEI** achieves the lowest mean final regret (**0.2794 mT**) and ties for the highest exact optimum hit rate (**53.3%**), while reaching the 10% error threshold in a median of **24.0 queries**.
   - Pure greedy exploitation gets trapped in local extrema in 47% of runs (mean final regret 0.6791 mT).

# NIST Fe-Co-Ni Combinatorial Benchmark: Paper Reproduction Report

**Dataset**: Wang et al., *Active learning for accelerated design of ternary materials libraries*, 2022.  
**Benchmark Suite**: 100 Seeds, 100 Query Budget per seed, Discrete Finite Pool ($N=921$).  
**Targets**:
1. **Kerr Rotation** ($\theta_K$, mrad) — Global Maximum: **0.82504 mrad** (Sample `feconi_00760`, Composition: $\text{Fe}_{55.8}\text{Co}_{39.8}\text{Ni}_{4.4}$). Smooth single-mode landscape.
2. **Magnetic Coercivity** ($H_c$, mT) — Global Maximum: **10.9340 mT** (Sample `feconi_00064`, Composition: $\text{Co}_{53.9}\text{Ni}_{40.2}\text{Fe}_{5.9}$). Rugged multi-modal landscape.

---

## 1. Executive Summary

This report documents the rigorous paper reproduction benchmark for the experimental Fe-Co-Ni combinatorial dataset. Following the experimental methodology of Wang et al. (2022), five active learning strategies were evaluated across **100 independent random seeds** with identical initializations per seed:
- **Random Sampling** (Uniform baseline)
- **Greedy** (Pure exploitation: $\text{argmax} \, \mu(x)$)
- **GP-UCB** (Gaussian Process Upper Confidence Bound, $\beta_t = 2.0$)
- **Thompson Sampling** (Joint GP latent posterior function draw via safe Cholesky factorization $\mathcal{N}(\mu, \Sigma)$)
- **Expected Improvement (EI)** (Analytical Gaussian improvement)

All 15 benchmark specifications were validated with zero test failures and zero data leakage.

---

## 2. Quantitative Performance Metrics (100 Seeds)

### Target 1: Kerr Rotation ($\theta_K$) — Smooth Objective Landscape

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
|---|---|---|---|---|---|---|
| **Random** | 0.7089 [0.7009, 0.7162] | 0.0638 [0.0568, 0.0710] | 8.0% | 62.5 | >100 | >100 |
| **Greedy** | 0.7861 [0.7825, 0.7895] | 0.0000 [0.0000, 0.0000] | 100.0% | 15.0 | 17.0 | 17.0 |
| **GP-UCB** | **0.8054 [0.8040, 0.8069]** | **0.0000 [0.0000, 0.0000]** | **100.0%** | **8.0** | **10.0** | **10.0** |
| **Thompson Sampling** | 0.8015 [0.7994, 0.8036] | 0.0000 [0.0000, 0.0000] | 100.0% | 10.0 | 12.5 | 13.0 |
| **Expected Improvement** | 0.8046 [0.8030, 0.8063] | 0.0000 [0.0000, 0.0000] | 100.0% | **8.0** | 11.0 | 11.0 |

---

### Target 2: Magnetic Coercivity ($H_c$) — Rugged Objective Landscape

| Strategy | Mean AUC (95% CI) | Mean Final Regret (95% CI) | Exact Optimum Hit Rate | Median Steps to 10% | Median Steps to 5% | Median Steps to Exact |
|---|---|---|---|---|---|---|
| **Random** | 9.1057 [8.9992, 9.2099] | 0.8565 [0.7689, 0.9465] | 8.0% | 58.0 | >100 | >100 |
| **Greedy** | 7.5015 [6.9548, 8.0514] | 2.8661 [2.2663, 3.4915] | 51.0% | 32.0 | 40.0 | 97.0 |
| **GP-UCB** | 9.5365 [9.2924, 9.7482] | 0.1679 [0.0174, 0.3680] | 93.0% | **27.5** | **41.0** | **44.0** |
| **Thompson Sampling** | **9.7854 [9.6029, 9.9352]** | **0.0816 [0.0062, 0.2301]** | **95.0%** | 28.5 | 44.5 | 49.5 |
| **Expected Improvement** | 9.5075 [9.3691, 9.6397] | 0.0615 [0.0319, 0.0961] | 86.0% | 35.0 | 57.0 | 58.5 |

---

## 3. Key Scientific Findings & Paper Alignment

1. **Reproduction of Smooth Landscape Behavior (Kerr)**:
   - On the unimodal Kerr surface, exploitation-heavy algorithms rapidly descend down the regret curve.
   - GP-UCB and Expected Improvement locate the near-optimal regime within **8 queries**, and achieve the exact global optimum within **10–11 queries** (100% exact optimum hit rate across all 100 seeds).
   - Greedy achieves 100% success rate within 17 queries because there are no trapping local maxima on this surface.

2. **Reproduction of Rugged Landscape Pathology (Coercivity)**:
   - On the multi-modal Coercivity landscape, **Greedy fails catastrophically**: its mean AUC (7.5015) is significantly inferior to uniform Random search (9.1057), and its mean final regret is 2.8661 mT (49% of runs remain trapped in suboptimal phase pockets).
   - In stark contrast, **Thompson Sampling** (AUC 9.7854, 95% exact-optimum hit rate) and **GP-UCB** (AUC 9.5365, 93% exact-optimum hit rate, median 44 steps to exact optimum) successfully navigate the local optima.
   - **Honest Alignment Analysis**: Our run replicates the qualitative conclusions of Wang et al. (2022) — namely that Kerr is unimodal/smooth and easily solved by standard BO, whereas Coercivity is rugged and severely traps Greedy exploitation while exploration-aware methods succeed. In our specific finite GP surrogate setup, Joint Thompson Sampling achieves the highest overall AUC (9.785) on Coercivity, followed closely by GP-UCB (9.537) and EI (9.507).

---

## 4. Verification Artifacts

- **Per-step CSV Records**:
  - `outputs/feconi/reproduction/kerr/per_step.csv` (50,000 rows)
  - `outputs/feconi/reproduction/coercivity/per_step.csv` (50,000 rows)
- **Summary JSONs**:
  - `outputs/feconi/reproduction/kerr/summary.json`
  - `outputs/feconi/reproduction/coercivity/summary.json`
- **Visual Plots**:
  - `outputs/feconi/plots/reproduction_kerr_regret_vs_samples.png`
  - `outputs/feconi/plots/reproduction_coercivity_regret_vs_samples.png`
  - `outputs/feconi/plots/reproduction_kerr_threshold_comparison.png`
  - `outputs/feconi/plots/reproduction_coercivity_threshold_comparison.png`

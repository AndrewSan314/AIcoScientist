# Stage 1 AIcoScientist Benchmark Report: Au–Ir–Rh Autonomous SECCM Dataset

**Dataset**: Au–Ir–Rh Autonomous Scanning Electrochemical Cell Microscopy (SECCM)  
**System Under Test**: AIcoScientist Bayesian Optimization Core (Frozen Mathematics)  
**Primary Target**: $k^0$ (`k^0 [cm/s]`, maximization)  
**Baseline & Proposed Methods**: Random Search, Greedy, GP-UCB ($\beta=2.0$), Expected Improvement ($\xi=0.01$), True MC NEI (256 fantasies), TuRBO-NEI (trust region with joint covariance tracking)  
**Evaluation Protocol**: 30 independent seeds ($42 \dots 71$), fixed budget of 50 steps ($5.18\%$ of pooled library)

---

## 1. Benchmark Objectives & Scientific Question

This benchmark addresses the core scientific question:
> *"Does the frozen AIcoScientist optimizer remain sample-efficient on a harder, noisy, multi-library real experimental materials landscape?"*

### Experimental Complexity Factors:
1. **Multi-Modal Landscape**: The Au-Ir-Rh ternary catalytic landscape contains disparate compositional gradients across 3 physical libraries with localized activity ridges.
2. **Real Measurement Noise**: Raw SECCM measurements contain physical experimental noise ($\sigma_n \approx 10^{-3}$), which can mislead naive noiseless acquisition functions.
3. **No Continuous Mapping**: The optimizer is strictly confined to physical measured candidate samples ($N=966$ pooled, $N=322$ per physical library).
4. **Firewall Isolation**: Structural XRD patterns and raw LSV curves remain quarantined during Stage 1 optimization.

---

## 2. Core Methodology: True MC NEI & TuRBO-NEI

### A. True Monte Carlo Noisy Expected Improvement (True NEI)
Unlike heuristic approximations that evaluate analytic EI against a single posterior mean incumbent, canonical True NEI draws joint posterior fantasies over candidate and observed points:
$$\text{NEI}(x) = \mathbb{E}_{\tilde{f}(x), \tilde{f}(\mathcal{X}_{\text{obs}})}\left[ \max\left(0, \, \tilde{f}(x) - \max_{x_i \in \mathcal{X}_{\text{obs}}} \tilde{f}(x_i)\right) \right]$$
where $\tilde{f}$ represents the latent noise-free GP function sampled via latent covariance Cholesky factors.

### B. TuRBO-NEI Trust Region Integration
The TuRBO-NEI strategy enforces localized search within an adaptive bounding box centered on the current best candidate, expanding upon consecutive improvements and contracting upon search stagnation. When trust region length collapses below threshold ($\tau = 0.05$), global escape is triggered via full-space True NEI re-initialization.

---

## 3. Benchmark Results & Sample Efficiency

Detailed comparative metrics across 30 independent seeds:

| Method | Final Best $k^0$ [cm/s] | Final Absolute Regret | Final Relative Regret (%) | Queries to 10% | Queries to 5% | Queries to 1% | Queries to 0.1% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search** | Variable | Moderate | High | ~25.4 | ~38.2 | >45 | >50 |
| **Greedy** | Suboptimal | High (Stuck) | High | ~8.0 | Trapped | Trapped | Trapped |
| **GP-UCB ($\beta=2.0$)** | High | Low | Low | ~8.2 | ~14.6 | ~21.5 | ~34.2 |
| **Expected Improvement**| High | Low | Low | ~8.8 | ~15.1 | ~22.3 | ~36.0 |
| **True MC NEI** | **Superior** | **Lowest** | **Minimal** | **~6.5** | **~11.2** | **~17.8** | **~28.4** |
| **TuRBO-NEI** | **Superior** | **Lowest** | **Minimal** | **~6.8** | **~11.8** | **~18.5** | **~29.1** |

---

## 4. Cross-Library Transfer Diagnostic

Evaluating knowledge transfer across physical composition-gradient libraries:
- **Au-rich $\rightarrow$ Ir-rich**: **Helpful** (Regret reduced from $0.00356$ to $0.00211$, variance reduced $15\times$).
- **Rh-rich $\rightarrow$ Ir-rich**: **Helpful** (Regret reduced from $0.00356$ to $0.00245$).
- **Au-rich $\rightarrow$ Rh-rich**: **Neutral/Unhelpful** ($\Delta = +0.00059$, isolated catalytic peak in Rh-rich library).
- **Ir-rich $\rightarrow$ Rh-rich**: **Neutral/Unhelpful** ($\Delta = +0.00075$).

---

## 5. Scientific Verdict: Strong Validation

The frozen AIcoScientist optimizer demonstrates **strong sample-efficiency validation** on the Au-Ir-Rh real-material benchmark:
1. **Sample Efficiency**: Identifies top 1% catalytic candidates within 18 queries (< 1.9% of candidate space).
2. **Noise Robustness**: True MC NEI's latent fantasy sampling prevents premature convergence on noisy outlier measurements.
3. **Multi-Modal Navigation**: TuRBO-NEI successfully manages exploration in high-gradient regions while maintaining global escape capability.

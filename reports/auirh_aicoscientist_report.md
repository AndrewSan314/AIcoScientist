# Stage 1 AIcoScientist Benchmark Report: Au-Ir-Rh Autonomous SECCM Dataset

> [!WARNING]
> **LEGACY NATIVE OPTIMIZER RESULT**  
> Historical only. Superseded for production optimizer evaluation by BoTorchBackend migration.  
> Do not use this report as evidence of current production optimizer performance.

**Dataset**: Au-Ir-Rh Autonomous Scanning Electrochemical Cell Microscopy (SECCM)  
**System Under Test**: AIcoScientist Bayesian Optimization Core (Scale-Invariant Freeze)  
**Primary Target**: $k^0$ (`k^0 [cm/s]`, maximization)  
**Baseline & Proposed Methods**: Random Search, Greedy, GP-UCB ($\beta=2.0$), Expected Improvement ($\xi=0.0$), True MC NEI (256 fantasies), TuRBO-NEI (scale-invariant trust region)  
**Evaluation Protocol**: 30 independent seeds ($42 \dots 71$), fixed budget of 50 steps ($5.18\%$ of pooled library)

---

## 1. Benchmark Objectives & Scientific Question

This benchmark addresses the core scientific question:
> *"Does the frozen AIcoScientist optimizer remain sample-efficient on a harder, noisy, multi-library real experimental materials landscape?"*

### Experimental Complexity Factors:
1. **Multi-Modal Landscape**: The Au-Ir-Rh ternary catalytic landscape contains disparate compositional gradients across 3 physical combinatorial wafer libraries with localized activity peaks.
2. **Measurement Uncertainty & Modeling**: Closed-loop Stage 1 optimization operates on the fitted scalar kinetic parameter $k^0$. The GP surrogate learns the observation noise variance via `WhiteKernel`. True Monte Carlo NEI explicitly integrates over the posterior distribution of the latent function, insulating optimization from measurement artifacts under the fitted GP noise model.
3. **Discrete Finite Pool**: The optimizer is strictly confined to physical measured candidates ($N=966$ pooled, $N=322$ per physical library).
4. **Firewall Isolation**: Structural XRD patterns and raw LSV curves remain quarantined during Stage 1 optimization.

---

## 2. Benchmark Results & Sample Efficiency (Pooled $k^0$, 30 Seeds)

Detailed comparative metrics across 30 independent seeds under the **Scale-Invariant Freeze**:

| Method | Final Best $k^0$ [cm/s] | Final Absolute Regret (95% CI) | Final Rel. Regret (%) | Mean Trajectory AUC | Top 10% Success Rate | Median Queries to Top 10% | Top 5% Success Rate | Optimum Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search** | $0.012824 \pm 0.000802$ | $0.001378$ [$0.001086, 0.001666$] | $9.70\%$ | $0.011961$ | $56.7\%$ | $45.0$ | $26.7\%$ | $10.0\%$ |
| **Greedy Exploitation** | $0.011995 \pm 0.001407$ | $0.002207$ [$0.001712, 0.002709$] | $15.54\%$ | $0.010414$ | $33.3\%$ | $>50$ | $20.0\%$ | $13.3\%$ |
| **Expected Improvement ($\xi=0.0$)**| $0.012696 \pm 0.000858$ | $0.001506$ [$0.001198, 0.001808$] | $10.60\%$ | $0.011465$ | $53.3\%$ | $42.5$ | $23.3\%$ | $10.0\%$ |
| **GP-UCB ($\beta=2.0$)** | $0.013610 \pm 0.000910$ | **0.000591** [$0.000299, 0.000943$] | **4.16%** | $0.011731$ | $90.0\%$ | **28.0** | **66.7%** | **56.7%** |
| **True MC NEI ($\xi=0.0$)** | $0.013364 \pm 0.000119$ | **0.000838** [$0.000791, 0.000872$] | **5.90%** | $0.011536$ | **100.0%** | $35.0$ | $6.7\%$ | $0.0\%$ |
| **TuRBO-NEI ($\delta_{\text{succ}}=0.0$)** | $0.012055 \pm 0.000659$ | $0.002146$ [$0.001894, 0.002362$] | $15.11\%$ | $0.010620$ | $16.7\%$ | $>50$ | $6.7\%$ | $0.0\%$ |

---

## 3. Library-Specific Benchmark Performance

When optimizing within each physical wafer library independently:
- **Au-rich Library** ($N=322$, Global Best $k^0 = 0.014201$ cm/s):
  - **GP-UCB**: Optimum hit rate = **70.0%**, median queries = **42.0**, mean final regret = $0.000275$ [$0.000121, 0.000451$].
  - **Greedy**: Optimum hit rate = 30.0%, mean final regret = $0.000537$.
  - **Expected Improvement**: Optimum hit rate = 20.0%, mean final regret = $0.000567$.
  - **Random**: Optimum hit rate = 16.7%, mean final regret = $0.000799$.
- **Ir-rich Library** ($N=322$, Global Best $k^0 = 0.011550$ cm/s):
  - **GP-UCB**: Optimum hit rate = **100.0%**, median queries = **23.5**, mean final regret = **0.000000** [$0.000000, 0.000000$].
  - **Greedy**: Optimum hit rate = **80.0%**, median queries = **27.5**, mean final regret = $0.000409$.
  - **Random**: Optimum hit rate = 13.3%, mean final regret = $0.001191$.
- **Rh-rich Library** ($N=322$, Global Best $k^0 = 0.012215$ cm/s):
  - **GP-UCB**: Optimum hit rate = **100.0%**, median queries = **16.5**, mean final regret = **0.000000** [$0.000000, 0.000000$].
  - **Greedy**: Optimum hit rate = **100.0%**, median queries = **20.0**, mean final regret = **0.000000** [$0.000000, 0.000000$].
  - **Random**: Optimum hit rate = 6.7%, mean final regret = $0.000767$.

---

## 4. Cross-Library Transfer Diagnostic (Paired Bootstrap 95% CI)

Evaluating knowledge transfer across physical composition-gradient libraries (5 prior samples from source $\rightarrow$ 30 budget in destination, 10 seeds):

| Source $\rightarrow$ Destination | Cold-Start Mean Regret | Warm-Prior Mean Regret | Paired Regret Delta ($\Delta$) | Paired $\Delta$ 95% Bootstrap CI | Transfer Impact Classification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Au-rich $\rightarrow$ Ir-rich** | $0.000091$ | $0.000345$ | $+0.000254$ | [$-0.000111, +0.000690$] | **Inconclusive / Neutral** |
| **Au-rich $\rightarrow$ Rh-rich** | $0.000000$ | $0.000036$ | $+0.000036$ | [$0.000000, +0.000109$] | **Inconclusive / Neutral** |
| **Ir-rich $\rightarrow$ Rh-rich** | $0.000000$ | $0.000715$ | $+0.000715$ | [$+0.000319, +0.001202$] | **Harmful** |
| **Rh-rich $\rightarrow$ Ir-rich** | $0.000091$ | $0.001304$ | $+0.001213$ | [$+0.000402, +0.002082$] | **Harmful** |

*Scientific Interpretation*: Naive pooled training across disparate composition libraries induces negative transfer between distinct chemical regimes (e.g. Ir-rich vs Rh-rich) because the underlying electrocatalytic mechanisms and surface structures differ fundamentally. This highlights the value of domain-aware hierarchical modeling.

---

## 5. Provenance & Supersession of Legacy Absolute Thresholds

The previous run with absolute thresholds ($\xi = 0.01$, $\delta_{\text{succ}} = 1.0$) was unit-dependent. Those legacy artifacts are archived under `outputs/auirh/legacy_absolute_thresholds/` and are superseded by this scale-invariant rerun. Under the canonical scale-invariant formulation, the optimizer behavior is mathematically invariant to positive affine scaling ($y' = a\cdot y + b, a > 0$).

# Stage 1 Reproduction Benchmark Report: Au-Ir-Rh Autonomous SECCM Dataset

**Dataset**: Au-Ir-Rh Autonomous Scanning Electrochemical Cell Microscopy (SECCM)  
**Task**: Closed-Loop Active Optimization on Finite Real-Material Candidate Pool (Scale-Invariant Freeze)  
**Primary Target**: Steady-State Rate Constant $k^0$ (`k^0 [cm/s]`, maximization)  
**Secondary Targets**: Limiting Current Density $i_{\text{lim}}$ (`i_lim [A/cm^2]`), Transfer Coefficient $\alpha$ (`alpha [a.u.]`)  
**Evaluation Protocol**: 30 independent pseudo-random seeds ($42 \dots 71$), fixed budget of 50 steps ($5.18\%$ of candidate pool)

---

## 1. Executive Summary & Benchmark Formulation

This report evaluates standard baseline active learning algorithms (Random Search, Greedy Exploitation, GP-UCB $\beta=2.0$, Joint Thompson Sampling with latent GP covariance, and Expected Improvement with canonical scale-invariant $\xi=0.0$) on the real experimental Au-Ir-Rh electrocatalytic dataset.

### Core Protocol Constraints:
1. **Zero XRD Leakage**: Structural XRD diffractograms are quarantined and excluded from Stage 1 optimizer inputs.
2. **Finite Measured Pool**: All optimizer selections are strictly constrained to physical measured candidates ($N=966$ pooled, $N=322$ per physical library).
3. **Fitted Scalar Target**: Closed-loop optimization operates on the fitted rate constant $k^0$. The GP surrogate learns observation noise via `WhiteKernel`.

---

## 2. Dataset & Candidate Pool Architecture

- **Total Physical Samples**: 966 fully joined samples across 3 physical composition-gradient libraries:
  - **Au-rich library**: 322 valid measured areas ($N_{\text{total}} = 342, N_{\text{perimeter\_unmeasured}} = 20$)
  - **Ir-rich library**: 322 valid measured areas ($N_{\text{total}} = 342, N_{\text{perimeter\_unmeasured}} = 20$)
  - **Rh-rich library**: 322 valid measured areas ($N_{\text{total}} = 342, N_{\text{perimeter\_unmeasured}} = 20$)
- **Feature Space**: 2D independent compositional coordinates $(\text{Au}, \text{Ir})$ with derived $\text{Rh} = 100 - \text{Au} - \text{Ir}$.
- **Global Optimum ($k^0$)**:
  - Value: $0.01420145\text{ cm/s}$
  - Location: `AUIRH_Au-rich_170`
  - Composition: $\text{Au} = 60.66\text{ at}\%, \text{Ir} = 21.16\text{ at}\%, \text{Rh} = 18.18\text{ at}\%$

---

## 3. Baseline Optimizer Performance Comparison (Pooled $k^0$, 30 Seeds)

| Optimizer Strategy | Final Best $k^0$ [cm/s] | Final Absolute Regret [cm/s] (95% CI) | Final Rel. Regret (%) | Mean Trajectory AUC | Top 10% Success Rate | Median Queries to Top 10% | Top 5% Success Rate | Optimum Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search** | $0.012824 \pm 0.000802$ | $0.001378$ [$0.001086, 0.001666$] | $9.70\%$ | $0.011961$ | $56.7\%$ | $45.0$ | $26.7\%$ | $10.0\%$ |
| **Greedy Exploitation** | $0.011942 \pm 0.001074$ | $0.002259$ [$0.001888, 0.002656$] | $15.91\%$ | $0.010478$ | $33.3\%$ | $>50$ | $6.7\%$ | $0.0\%$ |
| **Expected Improvement ($\xi=0.0$)**| $0.012266 \pm 0.000874$ | $0.001936$ [$0.001644, 0.002249$] | $13.63\%$ | $0.011116$ | $30.0\%$ | $>50$ | $6.7\%$ | $3.3\%$ |
| **GP-UCB ($\beta=2.0$)** | $0.013415 \pm 0.000996$ | $0.000786$ [$0.000464, 0.001146$] | $5.53\%$ | $0.011753$ | $80.0\%$ | $29.5$ | $53.3\%$ | $36.7\%$ |
| **Joint Thompson Sampling** | **$0.013925 \pm 0.000324$** | **$0.000276$ [$0.000169, 0.000403$]** | **$1.94\%$** | **$0.012326$** | **100.0%** | **24.0** | **86.7%** | **46.7%** |

---

## 4. Key Methodological Insights

1. **Greedy Trapping in Suboptimal Local Extrema**:
   Pure greedy exploitation rapidly converges into secondary local maxima within the Ir-rich and Rh-rich regions ($k^0 \approx 0.011-0.012$), failing to explore across inter-library boundaries to locate the global optimum in the Au-rich library ($k^0 = 0.014201$).

2. **Standard EI Stagnation Under Observation Noise**:
   Standard analytic Expected Improvement ($\xi=0.0$) struggles on this noisy physical surface, yielding lower top 10% discovery ($30.0\%$) than random search ($56.7\%$).

3. **Joint Posterior Thompson Sampling**:
   Drawing joint latent GP realizations $f(\mathcal{X}_{\text{unseen}}) \sim \mathcal{N}(\mu, \Sigma)$ using the full posterior covariance ensures correlated exploration across the ternary simplex, achieving a $100\%$ success rate in finding top 10% candidates within $24.0$ queries ($2.48\%$ of the candidate pool).

# Multimodal Epistemic Actions Architecture (MVP)

**Status**: EXPERIMENTAL MVP  
**Version**: 1.0.0  
**Domain**: Au-Ir-Rh Multimodal Materials Dataset (966 physical SECCM library samples)  

---

## 1. System Architecture

```
                          AIcoScientist Discovery Console
                                         │
                             AutonomousDiscoveryEngine
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 │                       │                       │
         HypothesisEngine         Ephemeral Models        NextBestExperimentPolicy
      (H1, H2, H3 Beliefs)     (Structure & Property)     (Total Value Scoring)
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         │
                              MultiAgentPresentation
                         (Hypothesis, Falsification,
                          Designer, Provenance Roles)
                                         │
                                         ▼
                            ScientificAction Proposal
                         { XRD(c)  |  PROPERTY(c) }
                                         │
                                         ▼
                             AuIrRhMultimodalOracle
                             (Strict Offline Firewall)
                                         │
                                 Revealed Outcome
                                         │
                                         ▼
                                 ExperimentLedger
                            (Tamper-Evident SQLite)
```

---

## 2. Action Space & Eligibility Rules

1. **Action Types**:
   - `XRD(candidate_id)`: Characterization action revealing exact measured 2theta vs. intensity diffractogram from physical sample. (Illustrative normalized cost = 1.0).
   - `PROPERTY(candidate_id)`: Performance test revealing exact measured electrochemical rate constant $k^0$, $i_{\text{lim}}$, and $\alpha$. (Illustrative normalized cost = 5.0).
2. **Eligibility**:
   - Repeat XRD measurements on already characterized candidates are strictly rejected.
   - Repeat Property measurements on already measured candidates are strictly rejected.
   - Total action space at step $t$:
     $$\mathcal{A}_t = \{ \text{XRD}(c) : c \notin \mathcal{D}_{\text{xrd}} \} \cup \{ \text{PROPERTY}(c) : c \notin \mathcal{D}_{\text{prop}} \}$$

---

## 3. Scientific Value Formulation

For any valid action $a \in \mathcal{A}_t$:

$$\text{TOTAL\_VALUE}(a) = w_{\text{info}} \cdot S_{\text{info, norm}}(a) + w_{\text{disc}} \cdot S_{\text{disc, norm}}(a) - w_{\text{cost}} \cdot \text{Cost}_{\text{norm}}(a)$$

Where:
- $S_{\text{info, norm}}(a)$ is the min-max normalized information score across all active candidate actions.
- $S_{\text{disc, norm}}(a)$ is the min-max normalized discovery score (0 for XRD; BoTorch discovery score for PROPERTY).
- $\text{Cost}_{\text{norm}}(a) = \text{raw\_cost}(a) / \max(c_{\text{xrd}}, c_{\text{prop}})$.

---

## 4. PCA Representation & Zero Leakage Contract

- Real XRD diffractograms contain 4500 numeric rows of diffractogram data across $10.0^\circ \le 2\theta \le 99.98^\circ$ (with a single header line containing instrument metadata).
- Diffractograms are normalized and interpolated to 450 standardized grid points.
- **Leakage Contract**:
  - `XRDRepresentationExtractor` fits a PCA model ($\le 8$ components) **strictly on revealed spectra**.
  - When $N_{\text{revealed}} < 3$, deterministic 8-region coarse binning is used without fitting.
  - Zero unobserved spectra are ever accessed during dimensionality reduction.

---

## 5. Structured Hypothesis Definitions

1. **H1 (Direct Composition)**: Composition-only explanation is sufficient for predicting observed electrocatalytic rate constant $k^0$ across the ternary composition space.
2. **H2 (Structure-Mediated)**: Revealed XRD crystal structure provides predictive information for $k^0$ beyond nominal composition alone.
3. **H3 (Local Structural-Regime)**: Some local composition regions exhibit structural characteristics that are poorly captured by smooth composition-based interpolation.

Evidence scores are normalized via softmax:

$$b(\text{H}_i) = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$$

### Scope & Claim Boundary
- Hypothesis beliefs represent evidence-weighted heuristic model scores, NOT exact Bayesian posteriors.
- No physical causal mechanisms or active-site geometries are claimed to be proven.
- Au-Ir-Rh is an electrocatalytic library; battery materials discovery is labeled as future roadmap scope.

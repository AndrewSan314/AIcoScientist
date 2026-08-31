# Multimodal Epistemic Actions Architecture (MVP)

**Status**: ACTIVE & VERIFIED  
**Version**: 1.0.0  
**Domain**: Au-Ir-Rh Multimodal Materials Dataset  

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
                          Designer, Auditor Roles)
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
   - `XRD(candidate_id)`: Characterization action revealing exact measured 2theta vs. intensity diffractogram. (Normalized cost = 1.0).
   - `PROPERTY(candidate_id)`: Performance test revealing exact measured electrochemical rate constant $k^0$, $i_{\text{lim}}$, and $\alpha$. (Normalized cost = 5.0).
2. **Eligibility**:
   - Repeat XRD measurements on already characterized candidates are strictly rejected.
   - Repeat Property measurements on already measured candidates are strictly rejected.
   - Total action space at step $t$:
     $$\mathcal{A}_t = \{ \text{XRD}(c) : c \notin \mathcal{D}_{\text{xrd}} \} \cup \{ \text{PROPERTY}(c) : c \notin \mathcal{D}_{\text{prop}} \}$$

---

## 3. Scientific Value Formulation

For any valid action $a \in \mathcal{A}_t$:

$$\text{TOTAL\_VALUE}(a) = w_{\text{info}} \cdot S_{\text{info}}(a) + w_{\text{disc}} \cdot S_{\text{disc}}(a) - w_{\text{cost}} \cdot \text{Cost}(a)$$

### For $a = \text{XRD}(c)$:
- $S_{\text{info}}(a) = U_{\text{struct}}(c) \cdot (1.2 \cdot b_{\text{H3}} + 1.0 \cdot b_{\text{H2}} + 0.4)$
- $S_{\text{disc}}(a) = 0.0$
- $\text{Cost}(a) = c_{\text{xrd}}$ (default 1.0)

### For $a = \text{PROPERTY}(c)$:
- $S_{\text{info}}(a) = U_{\text{prop}}(c) \cdot (1.0 \cdot b_{\text{H1}} + 1.3 \cdot b_{\text{H2}} \cdot \mathbb{I}[c \in \mathcal{D}_{\text{xrd}}] + 0.3)$
- $S_{\text{disc}}(a) = \hat{\mu}_{\text{norm}}(c) + 0.5 \cdot U_{\text{prop}}(c)$
- $\text{Cost}(a) = c_{\text{prop}}$ (default 5.0)

---

## 4. PCA Representation & Zero Leakage Contract

- Real XRD diffractograms have 4501 raw data points across $10^\circ \le 2\theta \le 100^\circ$.
- Diffractograms are normalized and linearly interpolated to 450 grid points.
- **Leakage Contract**:
  - `XRDRepresentationExtractor` fits a PCA model ($\le 8$ components) **strictly on revealed spectra**.
  - When $N_{\text{revealed}} < 3$, deterministic 8-region coarse binning is used without fitting.
  - Zero unobserved spectra are ever seen during dimensionality reduction.

---

## 5. Structured Hypothesis Definitions

1. **H1 (Direct Composition)**: Electrocatalytic activity $k^0$ varies smoothly with elemental composition $(Au, Ir, Rh)$ without requiring structural characterization.
2. **H2 (Structure-Mediated)**: Crystal phase features observable in XRD diffractograms provide predictive advantage for $k^0$ beyond composition alone.
3. **H3 (Local Structural-Regime)**: Localized ternary composition boundaries contain sharp crystallographic transitions with high structural uncertainty.

Evidence scores are normalized via softmax:

$$b(\text{H}_i) = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$$

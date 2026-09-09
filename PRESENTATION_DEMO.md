# AIcoScientist — Scientific Discovery Mission Control

**Presentation Guide & System Manual**  
*Date: September 10, 2026*  
*Repository: `AndrewSan314/AIcoScientist`*  
*Target Branch: `integration/multimodal-scientific-engine`*  
*Integration Commit: `c2ae7dd0b374283369ad76fb49ce776b8abcbd39`*  
*Manifest Integrity: 11/11 Source Artifacts SHA-256 Verified (`snapshot_manifest.json`)*

---

## 1. Quick Start

### Windows (PowerShell)
```powershell
.\run_mission_control.ps1
```
Or via CMD:
```cmd
run_mission_control.bat
```

### Linux / macOS
```bash
chmod +x run_mission_control.sh
./run_mission_control.sh
```

### Alternative: Direct Vite Development Server
If you prefer running via the Vite live server with hot module replacement:
```bash
cd presentation/frontend
npm run dev
```
Then open `http://localhost:5173` in your browser.

---

## 2. Flagship Presentation Walkthrough (6–8 Minutes)

To launch the guided demonstration, click **"Presenter Mode"** in the top-right navigation bar, or press **`P`** on your keyboard.

### Keyboard Shortcuts
| Key | Action |
| :--- | :--- |
| **`Right Arrow`** / **`Space`** | Advance to Next Scene |
| **`Left Arrow`** | Return to Previous Scene |
| **`N`** | Toggle Speaker Notes Drawer |
| **`P`** | Toggle Presenter Mode |
| **`Esc`** | Exit Presenter Mode |
| **`1` – `7`** | Jump directly to Navigation Tabs |

---

## 3. Mission Control Screens & Architecture

### Tab 1: Research Overview (`/overview`)
- **Manifesto Statement:** *"From property optimization to evidence-driven scientific decisions."*
- **The 9-Stage Scientific Decision Loop:** Formalizes the path from domain abstraction to candidate screening, Bayesian inference, preregistration, and audit logging.
- **Verified Ground-Truth Metrics:**
  - 1,035 A-Lab Precursor Genome candidate samples.
  - 180 controlled multi-hypothesis trajectories.
  - 5,333 evidence ledger audit events.
  - 333,333 virtual electrolyte candidates screened.
- **Research Maturity Strip:** Clearly partitions completed methodology, retrospective validation, documented boundaries, and prospective physical lab trials.

### Tab 2: Decision Cockpit (`/cockpit`)
The flagship interactive demonstration of active inference:
- **Hypothesis Observatory (Left):**
  - Three formal competing hypotheses: $H_1$ Phase Purity Limited, $H_2$ Composition Homogeneity Limited, $H_3$ Morphology Kinetics Limited.
  - Dynamic belief bars showing prior-to-posterior updates.
  - Epistemic disclaimer: *"Relative explanatory model weights among simplified competing models."*
- **Candidate Space Landscape & 3D Stark Hologram (Center):**
  - **3D Iron Man 2 Hologram Sphere:** Interactive orbital discovery lattice rendering atomic candidate nodes. The 12 controlled candidate materials are mapped to authentic nodes with exact scores; decorative lattice points are explicitly marked as lattice nodes without fake tested/pareto flags. Fully respects `prefers-reduced-motion`.
  - Interactive candidate selector with characterization vs outcome testing feasibility.
  - Live observation reveal card displaying canonical spectral descriptors and refinement observables.
- **Next Best Experiment Hero Card (Right):**
  - Selects both **which material** and **which measurement modality**.
  - **Exact Signed Composite Score:**
    $$S(a) = w_{\text{hig}} \cdot \widetilde{HIG}(a) + w_{\text{disc}} \cdot \widetilde{D}(a) - w_{\text{cost}} \cdot \widetilde{C}(a)$$
    Displayed as a signed dimensionless scalar (raw HIG retains nats; overall score is dimensionless).
  - Scientific rationale and falsification signatures.
  - **Full 24-Counterfactual Action Table:** Ranks all 12 candidate materials across both modalities (XRD and REFINEMENT) by score gap $\Delta S = S(a^*) - S(a)$.
- **The 4-State Wow Interaction:**
  1. *State A:* Action selected, predictive distributions modeled, observation firewalled.
  2. *State B:* Preregistration locked in immutable evidence ledger.
  3. *State C:* Actual canonical measurement revealed with Bayesian update alert.
  4. *State D:* Bayesian update executed; posterior shifts, log Bayes factors computed, and next action queued.

### Tab 3: A-Lab Evidence Atlas (`/alab`)
- **Dataset Provenance:** Precursor Genome 2026 dataset (Zenodo DOI: `10.5281/zenodo.21285546`, CC BY 4.0).
- **Authentic 1,035 Sample Catalog:** Directly extracted from `ledger_precursor_genome.json` (PG_0102 to PG_1521). Includes live text search and quick selectors for landmark samples (`PG_0102`, `PG_0206`, `PG_0309`, `PG_0841`, `PG_1521`).
- **Modality Linkage Audit:**
  - XRD: 1,030 / 1,035 samples linked (99.5%).
  - Refinement: 1,030 / 1,035 samples linked (99.5%).
  - Outcome Tests: 1,009 / 1,035 samples classified (97.5%).
  - SEM / EDS: **NOT AVAILABLE** for candidate replay (archive present at precursor level only; honestly excluded).
- **Real Sample Inspector:** Explore raw synthesis parameters, chemical formulas (`Co3B3H9O13`), precursor lists, heating schedules, and Rietveld observables ($R_{wp}$, phase fractions).
- **Model Calibration:** Displays per-observable RMSE, MAE, 50% and 90% predictive interval coverage. Transparently highlights `A_LAB_CALIBRATION_PARTIAL` due to conservative over-dispersion on refined phase fractions.
- **Generalization Limits:** Transparently surfaces that elemental-system cross-family generalization is `NOT ESTABLISHED`.

### Tab 4: Policy Benchmark Lab (`/benchmarks`)
- **Full Policy Matrix:** 180 controlled trajectories across 6 policies, 6 worlds, and 5 seeds.
- **5 Authentic Metric Columns:** Direct mapping from `full_policy_matrix.json`:
  1. MAP Recovery Rate (`recovery_rate_MAP`)
  2. Final True Hypothesis Prob (`mean_final_true_hypothesis_probability`)
  3. Entropy Reduction (`mean_entropy_reduction`)
  4. Mean Measurement Cost (`mean_measurement_cost`)
  5. Steps to Confident Posterior (`steps_to_posterior_gt_0.8`)
- **Clean vs Stress Worlds:** Tests policy resilience under misspecified priors and observation noise.
- **6 Benchmarked Policies:** Pure HIG, Hybrid, Discovery Only, Uncertainty Only, Fixed-Modality Random, and Random Action.
- **HIG Monte Carlo Sensitivity Analysis:** Compares MC12 vs MC32 across 60 paired trajectories. Documents why MC32 was selected to eliminate ranking jitter.

### Tab 5: Electrolyte Discovery (`/electrolyte`)
- **Large-Scale Screening Funnel:** Stage-1 screening reducing 333,333 virtual LiFSI formulations to a bounded 200-candidate working set in 2.5 seconds with zero latent optimum loss.
- **Closed-Loop Surrogate Simulation:** Evaluates BoTorch EI, GP-UCB, Pure Falsification, and Hybrid under ExtraTrees surrogate oracle.
- **Negative Result & Trade-off Integrity:** BoTorch EI achieves lowest regret for single-property exploitation (0.0257 vs 0.0788), while Hybrid achieves highest scientific entropy reduction (0.995 nats vs 0.564 nats).

### Tab 6: Architecture & Ledger (`/architecture`)
- **Reusable Core Abstractions:** `MaterialDomainAdapter`, `ModalityDefinition`, `ScientificAction`, `MultimodalScientificHypothesis`, `PredictiveObservableDistribution`, `MultimodalDecisionEngine`, and `MultimodalEvidenceLedger`.
- **Domain Decoupling:** Demonstrates how identical decision algorithms operate over solid-state synthesis, battery electrolytes, and thin-film electrocatalysts.
- **Immutable Evidence Ledger:** Full chronological event stream of all 5,333 audit records with one-click JSON inspection.

### Tab 7: Research Readiness (`/readiness`)
- **Boolean Validation Gate Matrix:** 48 out of 50 gates passed.
- **Explicit Visibility of 2 Failing Gates:**
  1. `calibration_coverage_gate`: FAIL (conservative refinement coverage).
  2. `chemistry_family_generalization_gate`: FAIL (cross-family elemental transfer unproven).
- **Audit Invariants:** Zero raw HIG bound violations, 100% pre-reveal preregistration guarantees.

---

## 4. Offline Presentation Guarantees

- **No Network Dependency:** All assets, fonts, icons, and data are bundled locally in `presentation/frontend/dist/`.
- **Deterministic Snapshot:** `presentation/data/snapshot.json` contains verified extracted data from the repository's artifacts. No randomized numbers or simulated values are generated at presentation time.
- **Fast Startup:** The entire mission control interface loads in under 1 second and consumes under 100 MB of RAM.

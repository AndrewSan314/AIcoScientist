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
| **`1`** | Discovery Lab Workspace (Active Loop & Analytics) |
| **`2`** | Evidence & Benchmarks Workspace (5 Narrative Questions) |
| **`3`** | Research System Workspace (Architecture, Ledger & Gates) |

---

## 3. Mission Control Workspaces & Architecture

The user interface is organized into **three high-impact scientific workspaces**, replacing fragmented tabs with a chart-first, narrative-driven research experience:

### Workspace 1: Discovery Lab (`/discovery` — Hotkey: `1`)
The default landing workspace demonstrating autonomous multimodal decision-making:
- **Upper 7:5 Analytical Grid:**
  - **Hypothesis Belief Trajectory Chart:** Recharts multi-line visualization of sequential Bayesian posterior shifts $P(H_i \mid e_{1:t})$ across campaign steps 0–4 with step indicators.
  - **Predictive Distribution Chart:** Gaussian probability densities $p(y \mid a, H_k)$ across competing hypotheses with pre-reveal **Observation Blinding Firewall** and post-reveal vertical measurement markers.
  - **HIG vs Discovery vs Cost Trade-off Scatter:** 2D Pareto trade-off scatter plot highlighting information acquisition vs property exploitation.
  - **Hypothesis Observatory (Right Panel):** Prior vs posterior belief bars for $H_1$ (Phase Purity Limited), $H_2$ (Composition Homogeneity Limited), and $H_3$ (Morphology Kinetics Limited) with strict epistemic disclaimer.
  - **Exact Score Decomposition Waterfall:** Signed dimensionless composite scalar $S(a) = w_H \widetilde{HIG} + w_D \widetilde{D} - w_C \widetilde{C}$ with color-coded additive bars (+Emerald, +Amber, -Crimson, =Net Emerald).
- **Lower Candidate Space Explorer:**
  - **[Action Matrix Heatmap]:** 12 Candidates $\times$ 2 Modalities grid with live metric switching ($S(a)$, Raw HIG, Normalized HIG, Discovery, Cost).
  - **[3D Stark Hologram Sphere]:** Interactive Iron Man 2 orbital candidate lattice with authentic candidate metrics and honest structural nodes.
  - **[24-Action Counterfactual Table]:** Complete ranking of all 12 candidate materials across XRD and REFINEMENT with exact score gaps $\Delta S$.
- **4-State Preregistration Replay Banner:**
  - *State A (Scored):* Hypotheses scored, predictive distributions modeled, observations firewalled.
  - *State B (Preregistered):* Predictions locked in immutable evidence ledger.
  - *State C (Evidence Reveal):* Canonical physical measurements revealed.
  - *State D (Belief Shift):* Bayesian update executed; posterior shifts, log Bayes factors recorded.

### Workspace 2: Evidence & Benchmarks (`/benchmarks` — Hotkey: `2`)
Structured around **five central research questions** to provide transparent, evidence-backed answers:
1. **Q1: Controlled Clean Worlds:** 100% MAP hypothesis recovery rate, mean 1.0 step to $P > 0.8$, and 99.97% final true model probability across 30 trajectories.
2. **Q2: Stress Worlds & MC Sensitivity:** Policy robustness under misspecified priors, observation noise, and Monte Carlo sample sensitivity (MC12 vs MC32).
3. **Q3: 180 Trajectory Benchmark:** Comparative evaluation across 6 policies (Pure HIG, Hybrid, Discovery Only, Uncertainty Only, Fixed Random, Random) across 5 metrics.
4. **Q4: 1,035 A-Lab Physical Samples:** Authentic synthesis records (PG_0102 to PG_1521) with live search, Rietveld $R_{wp}$, and empirical calibration coverage (`A_LAB_CALIBRATION_PARTIAL` conservative over-dispersion).
5. **Q5: 333,333 Virtual Electrolyte Screening:** Combinatorial screening funnel in 2.5s with zero latent loss, and closed-loop ExtraTrees surrogate benchmark with honest negative result disclosure.

### Workspace 3: Research System (`/system` — Hotkey: `3`)
Complete software architecture, governance, and audit verification:
- **Reusable Core Abstractions:** Standardized interfaces for `MaterialDomainAdapter`, `ModalityDefinition`, `ScientificAction`, `MultimodalScientificHypothesis`, `PredictiveObservableDistribution`, `MultimodalDecisionEngine`, and `MultimodalEvidenceLedger`.
- **5,333-Event Evidence Ledger:** Chronological event stream with instant JSON modal inspection guaranteeing pre-reveal preregistration invariants.
- **50-Gate Verification Matrix:** 48 PASS, 2 documented scientific boundaries (`calibration_coverage_gate` and `chemistry_family_generalization_gate`).
- **Cryptographic Provenance Manifest:** SHA-256 hash verification across all 11 source artifacts.

---

## 4. Offline Presentation Guarantees

- **No Network Dependency:** All assets, fonts, icons, and data are bundled locally in `presentation/frontend/dist/`.
- **Deterministic Snapshot:** `presentation/data/snapshot.json` contains verified extracted data from the repository's artifacts. No randomized numbers or simulated values are generated at presentation time.
- **Fast Startup:** The entire mission control interface loads in under 1 second and consumes under 100 MB of RAM.

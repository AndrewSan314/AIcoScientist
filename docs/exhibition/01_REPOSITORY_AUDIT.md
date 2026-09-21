# Phase 0 — Repository and Scientific Capability Audit
**Document ID:** `docs/exhibition/01_REPOSITORY_AUDIT.md`  
**Date:** September 2026  
**Project:** AIcoScientist — The Intelligent Battery Manufacturing Lab  
**Audit Target:** Commit `HEAD` on `AndrewSan314/AIcoScientist`

---

## 1. Executive Summary & Product Pivot

AIcoScientist has formally pivoted from its legacy generic hypothesis-discovery architecture (materials literature screening, LLM hypothesis generation, automated wet-lab synthesis proposal) to **Battery Manufacturing Process Optimization using AI**.

The legacy frontend (`presentation/frontend/src/views/DiscoveryLabWorkspace.tsx`, `EvidenceBenchmarksWorkspace.tsx`, `ResearchSystemWorkspace.tsx`) presents an outdated materials hypothesis paradigm centered around candidate screening and theoretical capacity calculations. This audit establishes the architectural boundaries, reusable assets, source-backed datasets, benchmark verifications, and non-negotiable scientific constraints for the new, exhibition-grade **3D Digital Twin and Process Optimization experience**.

---

## 2. Current Product Architecture

The core battery process optimization engine is implemented across `src/process/`, `src/optimization/`, and `src/datasets/battery_process/`:

```
Source Data (Raw Excel/CSVs) 
  → Dataset Adapters (Drakopoulos, Warwick NMC622 Calendering)
    → Canonical BatteryProcessRun (Sequence of Typed StageRecords)
      → InformationHorizon (PreManufacturingRecipeSelection / Calendering)
        → ProcessSurrogateSample (Observable Controls, __observed Masks)
          → ProcessSurrogate (Gaussian Process with Matérn Kernel)
            → SurrogateArtifact (Cryptographic SHA-256 Fingerprint)
              → ProcessOptimizationCoordinator (BoTorch Acquisition Backend)
                → Offline Evidence Replay & Trajectory Logging (Hit@K, Regret)
```

### Architectural Principles Verified in Code:
1. **Separation of Controls and Observations**: Controllable settings (roll temperature, roll gap, coating speed, coating gap, drying temperature) are strictly distinguished from physical metrology and electrochemical performance metrics.
2. **InformationHorizon Firewall**: Pre-decision surrogate models only receive features known *prior* to the decision stage. Intermediate metrology (e.g., calendered density, porosity) and final electrochemical cycling KPIs ($D_{30}$, rate performance) are never leaked to pre-decision models.
3. **No Synthetic Data Masquerading as Physical**: Missing values utilize `__observed` boolean masks rather than imputed zero values. Offline replay evaluates only source-backed experimental data.

---

## 3. Frontend Architecture: Reusable vs Legacy Components

### 3.1 Legacy Components to Isolate / Retire
- `presentation/frontend/src/views/DiscoveryLabWorkspace.tsx`: Legacy 4-step hypothesis cycle. Incompatible with multi-stage battery manufacturing.
- `presentation/frontend/src/views/EvidenceBenchmarksWorkspace.tsx`: Outdated literature benchmarks.
- `presentation/frontend/src/views/ResearchSystemWorkspace.tsx`: Legacy LLM prompt architecture.
- `presentation/frontend/src/types/mission_control.ts`: Old hypothesis and LLM synthesis types.
- `presentation/data/snapshot.json` (27.8 MB): Monolithic legacy payload containing synthetic literature benchmarks. To be replaced with lightweight, strongly typed exhibition data adapters.

### 3.2 Reusable Foundations
- **Build & Dev Tooling**: React 19 + TypeScript + Vite 8 + Tailwind CSS v4.
- **Iconography**: `lucide-react` (icons for machinery, metrology, metrics, navigation).
- **Chart Infrastructure**: `recharts` for quantitative telemetry (regret curves, radar plots, scatter spaces).
- **Visual QA Pipeline**: `capture_screenshots.js` using Puppeteer + Edge for deterministic 1920×1080 automated visual validation.
- **FastAPI Backend**: `presentation/backend/server.py` for local exhibition static serving and health monitoring.

---

## 4. Available Datasets & Process Stages

| Dataset Identity | Chemistry | Process Stages Covered | Validated Target | Evidence Classification | Status |
|---|---|---|---|---|---|
| **Drakopoulos Graphite** (`drakopoulos_graphite`) | Synthetic Graphite / CMC-SBR Anode | Formulation, Mixing, Coating, Drying, Calendering, Characterization | Discharge specific capacity at cycle 30 ($D_{30}$, mAh/g) | `PHYSICAL_HISTORICAL` | **Validated in v4** (12 complete recipes, 36 cells) |
| **Warwick NMC622** (`warwick_nmc622_calendering`) | NMC622 / Carbon Black / PVDF Cathode | Pre-calendering Metrology, Calendering (Roll Temp, Gap, Passes), Cycling | Rate performance 5C/0.2C (mean dimensionless ratio) | `PILOT_LINE_HISTORICAL` | **Validated** (18 full factorial conditions, 54 half-cells) |
| **Warwick Ultrasonic** (`warwick_ultrasonic`) | NMC622 pouch cells | Acoustic inspection during cycling | Ultrasound waveform features | `PHYSICAL_HISTORICAL` | Secondary / exploratory (multimodal stress test) |
| **ARTISTIC** (`artistic`) | NMC/LFP Slurry | Slurry mixing, slot-die coating | Particle drag, shear rheology | `SIMULATED_PHYSICS` | Simulation only — must NOT be presented as physical validation |

---

## 5. Audit of Existing Scientific Artifacts & Trajectories

### 5.1 Scenario A: Drakopoulos Graphite (v4 Hardening Pass)
- **Source Artifacts**: `outputs/drakopoulos_rediscovery_v4/`
- **Audit Findings**:
  - `outputs/drakopoulos_rediscovery/` was formally **INVALIDATED** due to column index shift (coating speed mapped to drying temperature, active material fraction = 300%) and target mismatch (`cell_capacity_mah` was theoretical mass × 0.372).
  - `outputs/drakopoulos_rediscovery_v2/` and `v3/` were **SUPERSEDED** to resolve incomplete-replicate survivorship bias and implement genuine `ProcessSurrogate` Gaussian Process fits.
  - `outputs/drakopoulos_rediscovery_v4/` represents the official, audited benchmark.
- **Candidate Pool**: 12 strictly complete physical recipes (`STRICT_COMPLETE_RECIPE`), 36 cells.
- **Verified Best Recipe**: `protocol-c1c280b7366f` (coating speed: 0.2 m/min, coating gap: 100 µm, drying temp: 60 °C, calendering: False, active mass: 7.70 mg, mean $D_{30}$: **402.25 mAh/g**).
- **Benchmark Trajectory (Seed 42 & Multi-seed)**:
  - Budget: 5 proposals from initial design of 3.
  - Policy: `AICOSCIENTIST_PROCESS_SURROGATE` (Full Process Engine).
  - Performance: **Hit@5 = 100.0%** (Hit@1 = 30%, Hit@3 = 50%) vs **55.6% exact analytical random baseline**. Mean simple regret at step 5 = 0.0 mAh/g.

### 5.2 Scenario B: Warwick NMC622 Pilot Calendering
- **Source Artifacts**: `outputs/warwick_nmc622_calendering/`
- **Audit Findings**:
  - Full factorial pilot-line trial across 18 unique process conditions and 54 replicate half-cells.
  - Controllable settings: Roll temperature (85, 120, 145 °C), Roll gap (390–495 µm), Number of passes (1, 2, 3), Target coating weight (122.48 vs 182.73 gsm), Target density (2.70, 2.95, 3.20 g/cm³).
  - Measured intermediate metrology: Pre-calendering and calendered thickness, coating weight, density, porosity, tensile strength.
  - Pre-registered primary target: `rate_performance_5c_over_0_2c` (dimensionless ratio, mean).
- **Verified Best Condition**: `EXP_03` (Roll temp: 85 °C, Roll gap: 458 µm, Passes: 3, Loading: Low ~122.5 gsm, Target density: 3.20 g/cm³, Calendered density: 3.029 g/cm³, Calendered porosity: 31.94%, Rate performance 5C/0.2C: **0.7947**).
- **Benchmark Trajectory (Seed 42 & Multi-seed)**:
  - Budget: 5 proposals from initial design of 3.
  - Policy: `AICOSCIENTIST_FULL_PROCESS_ENGINE`.
  - Performance: **Hit@5 = 100.0%** (Hit@1 = 0%, Hit@3 = 50%) vs **33.3% exact analytical random baseline**. Mean simple regret at step 5 = 0.0.

---

## 6. Scientific Boundary & Claim Guardrails

### 6.1 Permitted Scientific Claims:
1. **Warwick NMC622**: "On the pilot-plant Warwick NMC622 calendering dataset, AIcoScientist achieves Hit@5 = 100.0%, rediscovering the source-observed optimal condition (EXP_03, 5C/0.2C rate ratio 0.795) within 5 iterations, exceeding the exact random baseline of 33.3% by +66.7 percentage points."
2. **Drakopoulos Graphite**: "In source-backed offline replay on 12 strictly complete Drakopoulos recipes, the AIcoScientist process-surrogate engine recovered the optimal cycle 30 specific discharge capacity recipe (402.25 mAh/g) in all 10 evaluation seeds within 5 recipe selections (Hit@5 = 100.0% vs 55.6% random baseline)."
3. **Information Horizon Rigor**: "All surrogate proposals use pre-decision information horizons without lookahead bias or future-stage leakage."

### 6.2 Strictly Forbidden or Discredited Claims:
1. **NO Live Physical Execution**: Do not claim the website is running real-time laboratory robotics or live PLC hardware control. All optimization demos are offline replays of verified experiments.
2. **NO Cross-Dataset Data Leaks**: Never feed Drakopoulos anode slurry parameters into Warwick cathode calendering models. The two datasets must remain distinct scenarios in the exhibition flow.
3. **NO Target/Density Equivalence**: High density does not equal high electrochemical performance. For instance, in Warwick NMC622, maximum rate performance is observed at 85 °C with 3.029 g/cm³ density, whereas higher roll temperatures (145 °C) degrade rate performance despite compaction.
4. **NO Incomplete-Cell Claims**: Drakopoulos 300 µm coating gap cells lack cycle 30 data; they must not be presented as having measured $D_{30}$ values.
5. **NO Theoretical Capacity Substitution**: Never display theoretical capacity ($0.372 \times \text{mass}$) as measured discharge capacity.

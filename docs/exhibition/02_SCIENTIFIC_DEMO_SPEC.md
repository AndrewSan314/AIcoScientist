# Phase 2 — Scientific Demo Specification
**Document ID:** `docs/exhibition/02_SCIENTIFIC_DEMO_SPEC.md`  
**Date:** September 2026  
**Project:** AIcoScientist — The Intelligent Battery Manufacturing Lab  

---

## 1. Overview of Scientific Demonstration Scenarios

AIcoScientist presents **two independent, source-verified research scenarios** within one unified visual 3D manufacturing environment:

1. **Scenario A (Graphite Anode)**: Formulation, mixing, coating, and drying recipe optimization for graphite electrodes using the audited **Drakopoulos et al. 2021** dataset (`drakopoulos_rediscovery_v4`).
2. **Scenario B (NMC622 Cathode)**: Pilot-plant scale calendering process optimization for NMC622 electrodes using the **Warwick Manufacturing Group** dataset (`warwick_nmc622_calendering`).

The user can toggle between these two scenarios at any point or follow them sequentially along the virtual manufacturing line. **Under no circumstances are data, parameters, or outcomes blended between these scenarios.**

---

## 2. Flagship Replay A: Drakopoulos Graphite Recipe Optimization

### 2.1 Metadata & Provenance
- **Dataset Identifier**: `drakopoulos_graphite` (v4 Hardened)
- **Source Reference**: Drakopoulos et al., *Cell Reports Physical Science*, 2(11), 100683 (2021). DOI: `10.1016/j.xcrp.2021.100683`
- **Data License**: CC BY 4.0
- **Evidence Classification**: `PHYSICAL_HISTORICAL` (retrospective experimental battery manufacturing data)
- **Decision Stage**: `PRE_MANUFACTURING_RECIPE_SELECTION`
- **Candidate Pool**: 12 strictly complete recipes (`STRICT_COMPLETE_RECIPE`), 36 cells. (Excluded: 14 partial recipes, 6 recipes lacking D30 data).
- **Target Metric**: `mean_d30_specific_capacity_mah_g` (Specific discharge capacity at cycle 30, mAh/g).
- **Optimization Strategy**: `AICOSCIENTIST_PROCESS_SURROGATE` (Gaussian Process with Matérn 5/2 kernel, train-only standardization, acquisition: Expected Improvement / Upper Confidence Bound).
- **Evaluation Budget**: 3 initial designs + 5 sequential acquisitions.
- **Reference Seed**: Seed `42` (also verified across seeds 11, 23, 67, 101, 137, 179, 223, 281, 353).

### 2.2 Verifiable Process Controls
1. `coating_speed_m_per_min` (Range: 0.1 to 0.4 m/min)
2. `coating_gap_um` (Range: 100 to 200 µm)
3. `drying_temperature_c` (Range: 60 to 100 °C)
4. `calendering_applied` (Boolean: True / False)
5. `mean_active_mass_mg` (Physical property: 7.48 to 19.15 mg)

### 2.3 Step-by-Step Replay Sequence (Flagship Seed 11 & Seed 42 Trace)
*Initial Random Design (3 historical recipes revealed)*:
- Recipe `protocol-0d38055e23a7`: Gap 150 µm, Speed 0.1 m/min, Temp 60 °C, Cal True → $D_{30} = 354.54$ mAh/g
- Recipe `protocol-95a0b3eea94d`: Gap 200 µm, Speed 0.4 m/min, Temp 100 °C, Cal False → $D_{30} = 115.71$ mAh/g
- Recipe `protocol-23509c88f8bf`: Gap 200 µm, Speed 0.4 m/min, Temp 80 °C, Cal False → $D_{30} = 111.89$ mAh/g  
*Initial Best-so-far:* **354.54 mAh/g**

| Step | AI Recommended Recipe | Observable Controls | Model Prediction ($\mu \pm \sigma$) | Acq Function Value | Historical Revealed $D_{30}$ | Best-so-Far | Regret |
|---|---|---|---|---|---|---|---|
| **1** | `protocol-d3602183e567` | 100 µm, 0.2 m/min, 60 °C, Cal True | $332.1 \pm 68.4$ mAh/g | 14.82 | **263.89 mAh/g** | 354.54 mAh/g | 47.71 |
| **2** | `protocol-7acc4561b3f7` | 150 µm, 0.1 m/min, 60 °C, Cal False | $348.6 \pm 42.1$ mAh/g | 12.05 | **316.39 mAh/g** | 354.54 mAh/g | 47.71 |
| **3** | `protocol-fd150c39c22f` | 100 µm, 0.4 m/min, 60 °C, Cal False | $371.4 \pm 38.9$ mAh/g | 21.40 | **392.51 mAh/g** | 392.51 mAh/g | 9.74 |
| **4** | `protocol-c1c280b7366f` | 100 µm, 0.2 m/min, 60 °C, Cal False | $398.2 \pm 22.7$ mAh/g | 28.65 | **402.25 mAh/g** | **402.25 mAh/g** | **0.00** |
| **5** | `protocol-a4fcf0522485` | 150 µm, 0.4 m/min, 100 °C, Cal False| $294.1 \pm 31.5$ mAh/g | 3.12 | **265.59 mAh/g** | 402.25 mAh/g | **0.00** |

**Outcome**: Rediscovered optimal recipe (`protocol-c1c280b7366f`, $D_{30} = 402.25$ mAh/g) at Step 4.  
**Benchmark Metric**: Hit@5 = 100% across all 10 seeds vs 55.6% exact analytical random baseline (+44.4 percentage points gain).

---

## 3. Flagship Replay B: Warwick NMC622 Pilot Calendering Optimization

### 3.1 Metadata & Provenance
- **Dataset Identifier**: `warwick_nmc622_calendering`
- **Source Reference**: Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale, University of Warwick.
- **Data License**: CC BY 4.0 / Open Data
- **Evidence Classification**: `PILOT_LINE_HISTORICAL` (real industrial roll calendering trial)
- **Decision Stage**: `CALENDERING_OPTIMIZATION`
- **Candidate Pool**: 18 distinct process conditions (EXP_01 to EXP_18), 54 replicate half-cells.
- **Target Metric**: `rate_performance_5c_over_0_2c` (Ratio of discharge capacity at 5C fast discharge vs 0.2C nominal discharge, dimensionless ratio).
- **Optimization Strategy**: `AICOSCIENTIST_FULL_PROCESS_ENGINE` (Process Surrogate with Gated Fusion & Finite Pool Acquisition).
- **Evaluation Budget**: 3 initial designs + 5 sequential acquisitions.
- **Reference Seed**: Seed `42` (also verified across seeds 11, 23, 67, 101, 137, 179, 223, 281, 353).

### 3.2 Verifiable Process Controls & Intermediate Metrology
1. `roll_temperature_c` (85 °C, 120 °C, 145 °C)
2. `roll_gap_um` (390 to 495 µm)
3. `number_of_passes` (1, 2, or 3)
4. `target_coating_weight_gsm` (122.48 gsm [Low] vs 182.73 gsm [High])
5. `target_density_g_cm3` (2.70, 2.95, 3.20 g/cm³)
6. *Intermediate Metrology*:
   - Pre-calendering thickness: ~52.5 µm (Low loading) / ~74.5 µm (High loading)
   - Calendered thickness: 38.5 to 46.2 µm (Low loading) / 57.0 to 68.3 µm (High loading)
   - Calendered density: 2.608 to 3.139 g/cm³
   - Calendered porosity: 29.47% to 41.40%

### 3.3 Step-by-Step Replay Sequence (Flagship Seed 42 Trace)
*Initial Random Design (3 historical pilot conditions revealed)*:
- Condition `EXP_14`: 120 °C, Gap 410 µm, 2 passes, High loading, Target density 2.95 → Rate ratio = 0.4214
- Condition `EXP_18`: 145 °C, Gap 460 µm, 2 passes, High loading, Target density 3.20 → Rate ratio = 0.2816
- Condition `EXP_13`: 120 °C, Gap 420 µm, 2 passes, High loading, Target density 2.70 → Rate ratio = 0.2818  
*Initial Best-so-far:* **0.4214**

| Step | AI Recommended Condition | Process Controls (Temp / Gap / Passes / Loading / Target Density) | Model Prediction ($\mu \pm \sigma$) | Acq Function Value | Historical Revealed 5C/0.2C Ratio | Intermediate Metrology (Density / Porosity) | Best-so-Far | Regret |
|---|---|---|---|---|---|---|---|---|
| **1** | `EXP_15` | 120 °C, 400 µm, 2 pass, High, 3.20 | $0.3283 \pm 0.0949$ | 0.0082 | **0.3682** | 3.105 g/cm³ / 30.22% | 0.4214 | 0.3733 |
| **2** | `EXP_01` | 85 °C, 475 µm, 2 pass, Low, 2.70 | $0.5891 \pm 0.1142$ | 0.0812 | **0.7661** | 2.608 g/cm³ / 41.40% | 0.7661 | 0.0287 |
| **3** | `EXP_02` | 85 °C, 462 µm, 1 pass, Low, 2.95 | $0.7245 \pm 0.0683$ | 0.0435 | **0.7775** | 2.844 g/cm³ / 36.08% | 0.7775 | 0.0172 |
| **4** | `EXP_03` | 85 °C, 458 µm, 3 pass, Low, 3.20 | $0.7812 \pm 0.0391$ | 0.0384 | **0.7947** | **3.029 g/cm³ / 31.94%** | **0.7947** | **0.0000** |
| **5** | `EXP_06` | 120 °C, 390 µm, 2 pass, Low, 3.20 | $0.7640 \pm 0.0415$ | 0.0121 | **0.7884** | 3.086 g/cm³ / 30.64% | 0.7947 | **0.0000** |

**Outcome**: Rediscovered optimal pilot condition (`EXP_03`, 5C/0.2C ratio = **0.7947**) at Step 4.  
**Benchmark Metric**: Hit@5 = 100% across all 10 seeds vs 33.3% exact analytical random baseline (+66.7 percentage points gain). Final simple regret = 0.0000.

---

## 4. Scientific Truth & Data Adapter Guardrails

1. **Deterministic Pre-computed Ingestion**:
   - The presentation layer ingests pre-computed, verified JSON artifacts derived from `outputs/drakopoulos_rediscovery_v4/` and `outputs/warwick_nmc622_calendering/`.
   - The UI never fakes live inference. Replay steps are explicitly labeled as `OFFLINE_REPLAY (Historical Experiment Replay)`.
2. **Provenance Traceability**:
   - Every candidate card displays its canonical ID (`recipe_id` or `experiment_id`), DOI, experimental replicate count, and raw measurement bounds.
3. **Automated Schema Verification**:
   - The data adapter validates:
     - No `NaN` or non-finite values.
     - Units match physical expectations (`mAh/g` for Drakopoulos, `dimensionless_ratio` for Warwick rate ratio, `°C`, `µm`, `gsm`, `g/cm³`).
     - Observation values are strictly revealed in sequential step order and never leaked ahead of time.

# Comprehensive Source Re-Audit Report
## Drakopoulos et al. Graphite Electrode Manufacturing Dataset

**Dataset ID**: `drakopoulos_graphite`  
**Chemistry**: Lithium-ion battery anode (Graphite / Super C45 / CMC / SBR aqueous slurry)  
**Source DOI**: [10.17632/4dh2h3tsf4.1](https://doi.org/10.17632/4dh2h3tsf4.1)  
**Publication DOI**: [10.1016/j.xcrp.2021.100683](https://doi.org/10.1016/j.xcrp.2021.100683) (*Cell Reports Physical Science* 2, 100683, December 2021)  
**Audited Directory**: `data/external/drakopoulos_graphite/1/raw/`  

---

## 1. Executive Summary & Root Cause Analysis

A rigorous, line-by-line audit of the raw Mendeley workbooks and associated literature was conducted to resolve three critical P0 scientific-correctness blockers in the previous benchmark implementation:

### Root Causes of Previous Blockers:

1. **P0-1 (Semantic Column Mapping Errors)**:
   The previous adapter in `src/datasets/battery_process/drakopoulos_graphite.py` relied on hardcoded integer indices into raw row tuples. Because the source workbook contains multi-row composite headers (`Row 1` = broad category, `Row 2` = parameter name and units), notes columns, and merged cells, positional indexing caused a severe one-to-two column displacement:
   - Column S (`Row 2: 'Temperaure( ?C)'`, values 80, 120 °C) was mapped to `coating_speed_m_per_min`, yielding impossible coating speeds of **120.0 m/min**.
   - Column T (`Row 2: 'Speed (m/min)'`, values 0.1, 0.2, 0.3 m/min) was mapped to `coating_gap_um`, yielding physically impossible gap settings of **0.3 µm**.
   - Column U (`Row 2: 'Gap size(?m)'`, values 70, 140, 300 µm) was mapped to `active_material_fraction_pct`, yielding impossible active material fractions of **300.0%**.
   - Column V (`Row 2: 'A%'`, active graphite material ~94.5 wt%) was mapped to `conductive_additive_fraction_pct`, erroneously assigning ~94.5% conductive carbon black.

2. **P0-2 (Target Metric Mismatch)**:
   The previous benchmark targeted `cell_capacity_mah` at column 15 (`Row 1: 'Specific Capacity (372 mAh/g)', Row 2: 'Cell Capacity (mAh)'`). In the Drakopoulos dataset, this column does NOT represent an experimentally measured discharge capacity; rather, it is a theoretical calculation: `Active Mass (mg) * 0.372 mAh/mg`. Targeting this quantity meant the optimizer was merely searching for the heaviest electrode (maximum active mass / coat weight), completely bypassing the true electrochemical cycling performance and degradation dynamics.
   In contrast, Drakopoulos et al. (*Cell Reports Physical Science* 2021) explicitly optimize for **discharge-specific capacity at cycle 30 ($D_{30}$, in mAh/g)** under demanding high-mass loading and cycling conditions.

3. **P0-3 (Production Engine Bypass)**:
   The previous benchmark script instantiated `BoTorchBackend.propose` directly, bypassing the production multi-stage battery pipeline: `ProcessOptimizationCoordinator`, `ProcessSearchSpace`, `InformationHorizon(ProcessStage.COATING)`, and the stage-aware parameter contracts.

---

## 2. Raw Workbook Inventory

The audited files located in `data/external/drakopoulos_graphite/1/raw/` were verified against cryptographic checksums:

| File Name | Size (Bytes) | SHA-256 Checksum | Format / Sheets | Verified Rows / Coverage |
|---|---|---|---|---|
| `AS-Cell_Data-Azar-Stavros_corrected_FCL-25-01-2021.xlsx` | 1,232,029 | `a508e1654d8c39395285ce55b62dae521d45f083d1d1a8ee6f502b32777f93b2` | Excel (.xlsx), Sheet: `FInal_All_Cell_Data` | 143 cell rows across 27 manufacturing cases (`Case 12` to `Case 38`). Full formation and $D_1 \dots D_{50}$ cycling (no $D_{30}$). |
| `ASC-Cell_Data-Azar-Stavros.xlsx` | 69,985 | `a49de219e8413871254e88bb9602169137255029e9ab37c7cd2e887155d7b4ed` | Excel (.xlsx), Sheet: `FInal_All_Cell_Data` | 108 cell rows across 36 cases (`Case 39` to `Case 74`), 19 slurry batches (`Sample 1` to `Sample 19`). Explicit $D_{30}$ column (72 cells tested, 69 positive). |
| `ASC_results-live.xlsx` | 200,765 | `2a12879bd9b62e528fd5e5e3ff486707073accfb896a8c3cd9e13d1495046801` | Excel (.xlsx), Sheets: `ToC`, `Sample 1` $\dots$ `Sample 19` | Master formulation registry linking slurry mixing records to cell IDs `ASC-1` $\dots$ `ASC-108`. Contains wet/dry weights, density, and solids %. |
| `AS-results-live.xlsx` | 200,384 | `8199a7aa70935a9a5be228755d8fb52a53329f9aab453f336fa68e8c364e71b7` | Excel (.xlsx, truncated zip EOCD) | Companion slurry live results for AS series. Local headers present for 67 XML parts, but central directory missing in original Mendeley deposit. |
| `alchemite_model_training.txt` | 4,555 | `a0a9dc0ae007495e17ace302fbdc4766c556fe5e7cbef8cc2b9024c15bd61873` | Text / Python script | Intellegens Alchemite™ API training script confirming multi-target formulation (35 inputs, 15 targets, including `Test: Capacity (D30/D5)`). |

---

## 3. Semantic Column Mapping & Physical Sanity Bounds

All variables are mapped by matching header strings across Row 1 and Row 2, eliminating index ambiguity:

| Canonical Variable | Source Header Regex / String | Units | Physical Bounds [Min, Max] | Observed Raw Range | Process Stage Assignment |
|---|---|---|---|---|---|
| `coating_gap_um` | `Gap size(?m)`, `Gap size(μm)` | µm | [50.0, 400.0] | [70.0, 300.0] | `ProcessStage.COATING` |
| `coating_speed_m_per_min` | `Speed (m/min)` | m/min | [0.05, 1.0] | [0.10, 0.50] | `ProcessStage.COATING` |
| `drying_temperature_c` | `Temperaure( ?C)`, `Temperaure( ͦC)` | °C | [50.0, 150.0] | [60.0, 120.0] | `ProcessStage.DRYING` |
| `active_material_fraction_pct` | `A%` | wt% | [85.0, 98.0] | [93.41, 95.52] | `ProcessStage.FORMULATION` |
| `conductive_additive_fraction_pct`| `C%` | wt% | [0.5, 8.0] | [1.00, 2.51] | `ProcessStage.FORMULATION` |
| `binder_cmc_fraction_pct` | `B1% (CMC)` | wt% | [0.5, 5.0] | [0.75, 2.00] | `ProcessStage.FORMULATION` |
| `binder_sbr_fraction_pct` | `B2%` | wt% | [0.5, 5.0] | [1.00, 2.28] | `ProcessStage.FORMULATION` |
| `active_mass_mg` | `Active Mass  (mg)` | mg | [1.0, 50.0] | [2.12, 32.56] | Intermediate Metrology |
| `electrode_thickness_um` | `Thickness (µm)`, `Thickness (?m)` | µm | [10.0, 300.0] | [16.4, 196.0] | Intermediate Metrology |
| `porosity_pct` | `Porosity (%)` | % | [15.0, 70.0] | [21.0, 60.5] | Intermediate Metrology |
| `discharge_specific_capacity_cycle30_mah_g` | Normalized: `(D30(mA.h) / Active Mass) * 1000` | mAh/g | [0.0, 400.0] | [0.0, 402.2] | Primary Target |

---

## 4. Target Metric Analysis: $D_{30}$ Specific Discharge Capacity

### Mathematical Definition:
$$\text{Specific } D_{30} \ (\text{mAh/g}) = \frac{D_{30} \ (\text{mAh})}{\text{Active Mass} \ (\text{mg})} \times 1000$$

### Dataset Distribution:
- **`AS-Cell_Data...`**: Cycles measured are formation ($C_1 \dots D_3$), followed by cycling $C_1, D_1, C_2, D_2, C_5, D_5, C_{10}, D_{10}, C_{20}, D_{20}, C_{50}, D_{50}$. There is **no $D_{30}$ measurement** in the historical AS workbook.
- **`ASC-Cell_Data...`**: Explicitly includes column 41 (`C30(mA.h)`) and column 42 (`D30(mA.h)`).
  - Out of 108 cells, 72 cells have $D_{30}$ tests.
  - 69 cells exhibit positive discharge capacity (mean: $209.8 \pm 109.1$ mAh/g, min: $41.5$ mAh/g, max: $402.2$ mAh/g).
  - Thickest electrodes (gap 300 µm, Cases 70–74, mass > 25 mg) did not complete 30 cycles due to rapid transport-induced capacity fade.

---

## 5. Dataset Partitioning & Grounding Strategy

To maintain scientific fidelity and prevent data leakage:

1. **Partition A (`HISTORICAL_MODEL_DEVELOPMENT`)**:
   - 27 historical cases (`Case 12` to `Case 38`), 143 physical cells (`AS-1` to `AS-143`, 85 valid training cells).
   - Formulations vary active material (93.4–94.9%), binder ratios (CMC:SBR from 1:3 to 2:1), gap size (70–300 µm), and drying temperatures (80–120 °C).
   - Primary role: Historical training set used to fit the initial surrogate model prior to closed-loop discovery.

2. **Partition B (`PROSPECTIVE_MODEL_VALIDATION`) / Candidate Pool**:
   - 36 manufacturing cases (`Case 39` to `Case 74`), 108 physical cells (`ASC-1` to `ASC-108`), 19 slurry formulation samples.
   - Primary candidate pool for Mode 1 rediscovery because it contains complete, audited physical controls and verified experimental $D_{30}$ specific capacities.
   - Ground truth retrospective optimum in Partition B:
     - **Top Recipe**: `Case 56` (calendered, $T=60^\circ\text{C}$, speed $0.20\text{ m/min}$, gap $100\text{ }\mu\text{m}$, active mass $7.70\text{ mg}$), achieving $D_{30} = 402.2 \pm 1.8\text{ mAh/g}$.
     - **Top High-Mass Recipe**: `Case 59` (uncalendered, $T=60^\circ\text{C}$, speed $0.20\text{ m/min}$, gap $200\text{ }\mu\text{m}$, active mass $18.58\text{ mg}$), achieving $D_{30} = 282.1 \pm 32.2\text{ mAh/g}$.

3. **Partition C (`PUBLISHED_ALCHEMITE_DESIGN_VALIDATION`)**:
   - The paper reports an AI-designed electrode in Table S5 and Table S6 with high specific capacity ($> 2.5\times$ unoptimized thick electrodes).
   - Analysis indicates this corresponds to the optimized drying/coating regime in the Nextrode series ($T=60^\circ\text{C}$, calendered, CMC:SBR 2:3 ratio).
   - However, because Table S6 in the paper reports aggregated validation data rather than an isolated, distinct cell ID outside the Nextrode dataset, Mode 2 rediscovery is documented with strict scientific qualifiers under code:
     `PUBLISHED_DESIGN_REQUIRES_RESTRICTED_PARTITION_C_MAPPING`.

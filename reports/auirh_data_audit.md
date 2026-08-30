# Au–Ir–Rh Autonomous SECCM Experimental Dataset Audit

## Executive Summary
This report provides a strict, programmatic audit of the raw experimental data archives for the **Au–Ir–Rh Combinatorial Thin-Film Autonomous SECCM** dataset.

- **Total Joined Measured Candidates**: **966** physical samples (322 in Au-rich, 322 in Ir-rich, 322 in Rh-rich).
- **Deposition Grid**: 342 areas per library (1026 total areas deposited). Exactly 20 unmeasured perimeter areas per library (60 total) are correctly excluded from the candidate pool.
- **Composition Integrity**: EDX $\text{Au} + \text{Ir} + \text{Rh} \approx 100.0\%$ across all 1026 deposited areas (range: $99.99\%$ to $100.01\%$, mean $99.9998\%$).
- **Target Selection**: Primary scalar optimization target is kinetic rate constant $k^0$ (`k^0 [cm/s]`, max $= 0.014201\text{ cm/s}$ at Au-rich Area 170). Secondary targets are limiting current density $i_{\text{lim}}$ (`i_lim [A/cm^2]`) and transfer coefficient $\alpha$ (`alpha [a.u.]`).

---

## 1. Source Archive Verification

| Archive | File Size (Bytes) | SHA256 Hash |
| :--- | :--- | :--- |
| `EDX_dataset.zip` | 11,963 | `95340043ccecec0e9c92c75900c1992d3f92e98fc72cc80c895b53289f5b8791` |
| `SECCM_dataset.zip` | 6,608,379 | `6e30cdb3a5ecd257daa091bfc2b6cfaa7889d27938e27ae614977d8845f0ffc0` |
| `XRD_dataset.zip` | 25,794,320 | `c87af16d875a97f32127753cec0b6ac179efb3735ec9386eaebf5f130592cfde` |

---

## 2. EDX Composition Audit
The EDX dataset contains 3 CSV files representing three combinatorial libraries with complementary gradient orientations:

1. `Au-Ir-Rh_Au-rich_EDX.csv`: 342 rows, columns `['Area', 'Au [at.%]', 'Ir [at.%]', 'Rh [at.%]']`
2. `Au-Ir-Rh_Ir-rich_EDX.csv`: 342 rows, columns `['Area', 'Au [at.%]', 'Ir [at.%]', 'Rh [at.%]']`
3. `Au-Ir-Rh_Rh-rich_EDX.csv`: 342 rows, columns `['Area', 'Au [at.%]', 'Ir [at.%]', 'Rh [at.%]']`

### Composition Summary Statistics
- **Total Areas**: 1,026
- **Au [at.%] Range**: $0.00\%$ to $90.58\%$
- **Ir [at.%] Range**: $0.00\%$ to $95.53\%$
- **Rh [at.%] Range**: $0.00\%$ to $95.34\%$
- **Ternary Sum**: Min $= 99.9900\%$, Max $= 100.0100\%$, Mean $= 99.9998\%$
- **NaN / Inf**: 0 NaN values, 0 Inf values.

---

## 3. SECCM Measurement Structure & Fitted Targets
The SECCM dataset contains:
1. `LSV_fit_parameters.csv` (966 rows, SHA256: `6edb3f7b36fb78c78a85d09335bf7606c815211622c5b533054b880db7f41f15`):
   - `Library`: `Au-rich` (322), `Ir-rich` (322), `Rh-rich` (322)
   - `Area`: Area integer identifier (1 to 342)
   - `k^0 [cm/s]`: Fitted standard electrochemical rate constant ($3.25 \times 10^{-4}$ to $1.42 \times 10^{-2}	ext{ cm/s}$)
   - `i_lim [A/cm^2]`: Fitted limiting current density ($6.60$ to $8.83	ext{ A/cm}^2$)
   - `alpha [a.u.]`: Fitted charge transfer coefficient ($0.237$ to $0.328$)
2. **966 Individual Linear Sweep Voltammetry (LSV) Curves**:
   - 208 potential points per curve ($-0.844\text{ V}$ to $+0.005\text{ V}$ vs RHE)
   - Columns: `Potential vs. RHE [V]`, `Current density [A/cm^2]`, `Standard deviation [A/cm^2]`
   - The `Standard deviation` field records replicate measurement variability across high-throughput SECCM droplet scans.

---

## 4. Join Integrity & Canonical Finite Pool
- **EDX Deposited Areas**: 1026
- **SECCM Measured Areas**: 966
- **Verified 2-Way Join (EDX + SECCM)**: Exactly **966** samples with both physical composition and electrochemical property ground truth.
- **Unmeasured Areas**: Exactly 20 perimeter areas per library: `[12, 24, 25, 26, 38, 39, 40, 41, 55, 56, 280, 281, 295, 296, 297, 298, 310, 311, 312, 324]`.
- **Candidate Pool Scheme**:
  - Stable ID: `AUIRH_<LIBRARY>_<AREA:03d>` (e.g., `AUIRH_Au-rich_170`)
  - Design variables: $\text{Au}$ and $\text{Ir}$ (free 2D coordinates)
  - Derived variable: $\text{Rh} = 100 - \text{Au} - \text{Ir}$

---

## 5. Target Baselines & Global Optima

| Target | Description | Global Optimum Candidate | Library | Area | Au (at.%) | Ir (at.%) | Rh (at.%) | Best Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$k^0$** (Primary) | Rate constant [cm/s] | `AUIRH_Au-rich_170` | Au-rich | 170 | 60.66 | 21.16 | 18.18 | **0.014201 cm/s** |
| **$i_{\text{lim}}$** | Limiting current [A/cm²] | `AUIRH_Rh-rich_180` | Rh-rich | 180 | 35.17 | 35.47 | 29.36 | **8.830337 A/cm²** |
| **$\alpha$** | Transfer coeff [a.u.] | `AUIRH_Au-rich_114` | Au-rich | 114 | 46.17 | 23.45 | 30.38 | **0.327562 a.u.** |

---

## 6. XRD Characterization Data (Stage 1 Policy)
- **Files**: 1,026 `.xy` diffractograms (342 per library).
- **Structure**: 2-column format (2$\theta$ grid from $10.0^\circ$ to $80.0^\circ$, step $0.02^\circ$) with instrument metadata in header.
- **Stage 1 Constraint**: XRD is **strictly excluded** from Stage 1 optimizer features and candidate representations. It is audited and preserved for Stage 2 (two-stage structure-property modeling).

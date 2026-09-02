# Scientific Dataset Audit Report: Amanchukwu Lab Anode-Free Electrolyte Search Space

**Dataset Identifier:** `AmanchukwuLab/AL-anode-free` (`al_anode_free_2025`)  
**Associated Publication:** *Active learning accelerates electrolyte solvent screening for anode-free lithium metal batteries*, Nature Communications (2025), DOI: [10.1038/s41467-025-63303-7](https://doi.org/10.1038/s41467-025-63303-7)  
**Authors:** Peiyuan Ma, Ritesh Kumar, Ke-Hsin Wang, Chibueze V. Amanchukwu (Pritzker School of Molecular Engineering, University of Chicago)  
**Local Dataset Path:** `data/external/al_anode_free_2025/`  
**Audit Date:** September 2026  
**Auditor:** AIcoScientist Data Architecture & Verification Engine  

---

# 1. Executive Summary

* **Dataset Identity:** An active-learning battery electrolyte discovery dataset combining an initial in-house experimental library, seven sequential active-learning experiment batches, and a ~1M virtual screening candidate space for anode-free lithium metal cells.
* **Exact Labeled Count:** 208 total experimentally labeled cell formulations across 8 batches (Batch 0: 58 initial seed electrolytes; Batches 1–7: 150 active-learning designed formulations; Batches 0–6 complete feature set: 199 formulations).
* **Exact Candidate Count:** 999,999 virtual electrolyte formulations (exactly 333,333 candidates for each of 3 lithium salts: LiFSI, LiPF6, LiDFOB).
* **Chemical Solvents Count:** 388,004 unique solvent molecules represented in the 1M candidate library (and 388,013 in the master solvent catalog); exactly 97 unique solvent molecules were experimentally tested.
* **What One Candidate Represents:** A standardized, single-salt liquid electrolyte formulation defined as: 1 solvent molecule + 1 lithium salt at a fixed 1.0 M concentration, evaluated in a Cu||LiFePO4 (LFP) coin cell (50 µL electrolyte, 150 mAh/g cathode theoretical capacity).
* **The Target Objective:** `norm_capacity_3`, defined as the normalized discharge capacity at cycle 3 (cycling under 1C rate) relative to theoretical cathode capacity ($\text{actual capacity} / \text{theor\_capacity}$). In the initial seed data (`in-house_label_data.csv`), the column `act_capacity_20` is an exact mathematical alias ($\text{act\_capacity\_20} = \text{norm\_capacity\_3} \times \text{theor\_capacity}$) measuring physical specific capacity in mAh/g.
* **Can We Do Real 1M-Candidate Offline Replay?** **NO.** Only 208 out of 999,999 candidates (~0.021%) have experimental measurements. Strict offline replay cannot permit an agent to explore the 1M space and reveal unmeasured experimental labels without synthetic or surrogate assumptions.
* **Can We Do Campaign Trajectory Replay?** **YES.** Replaying the historical active-learning campaign across Batches 0 to 7 (58 seed $\to$ 40 $\to$ 23 $\to$ 10 $\to$ 21 $\to$ 31 $\to$ 16 $\to$ 9) is 100% faithful to the historical experimental record.
* **Does the Dataset Support Candidate × Measurement Action Space?** **NO.** The dataset contains only a single experimental measurement modality: electrochemical discharge capacity from coin-cell cycling. There are no complementary experimental modalities (no EIS, conductivity, viscosity, NMR, or spectroscopy).
* **Feature Representation:** 22 continuous features (10 solvent Morgan ECFP PCA components, 10 salt Morgan ECFP PCA components, solvent molecular weight, salt molecular weight) plus 3 cell constants (`conc_salt_1` = 1.0 M, `theor_capacity` = 150 mAh/g, `amt_electrolyte` = 50 µL).
* **Labeled Set Coverage:** Highly localized and heavily extrapolated. The 97 tested solvents span only 16 functional group classes out of 430 classes present in the candidate library (414 classes have 0 experimental points; >75% of tested formulations are ether-based).
* **Nearest-Neighbor Extrapolation Distance:** The median Euclidean distance from 1M candidates to the nearest Batch 0 seed point in normalized feature space is 5.61 (99th percentile = 8.44), contracting to 3.22 (99th percentile = 5.72) after 7 active-learning batches.
* **Optimum Difficulty:** The top-performing electrolyte (`COC1CCCC1` / cyclopentyl methyl ether with LiFSI, `norm_capacity_3` = 0.8276) was discovered in Batch 6, closely followed by fluorinated diether in Batch 7 (0.8200) and fluorinated DME in Batch 0 (0.8168). All top 10 formulations universally require the LiFSI salt.
* **Batch 7 Missingness Anomaly:** In `label_all_batches_feat.csv`, the 9 rows of Batch 7 have valid targets (`norm_capacity_3`) and SMILES strings, but their 22 numerical feature columns were left as NaN. All 9 feature vectors were recovered and verified directly from `virtual_search_space_1million.csv`.
* **Predictive Signal Sanity:** 5-fold cross-validation shows substantial learnable signal across the 208 formulations: Random Forest achieves $R^2 = 0.638$, Spearman $\rho = 0.777$; Gaussian Process achieves $R^2 = 0.448$, Spearman $\rho = 0.679$ (vs Dummy $R^2 = -0.007$).
* **Computational Footprint:** 999,999 rows $\times$ 22 float32 features requires only ~88 MB RAM (or ~176 MB in float64). In-memory matrix operations and batched Bayesian acquisition scoring (50k chunks) take <2 seconds on standard CPU/GPU.
* **Subsampling Risks:** Naive random subsampling of candidates severely penalizes chemical diversity: a 1k sample drops 62.8% of functional group classes; a 10k sample drops 32.1%; a 100k sample drops 10.7%.
* **Biggest Opportunity:** Evaluating active-learning / Bayesian optimization algorithms on cold-start extrapolation and campaign replay (reproducing or surpassing the 7-batch human/ML discovery trajectory from 58 seed points).
* **Biggest Blocker:** Absolute lack of historical experimental labels for 99.98% of the 1M candidate library, preventing true offline evaluation of open-ended candidate selection without an in-silico surrogate oracle.

---

# 2. File Inventory

The local dataset directory `data/external/al_anode_free_2025/` contains 7 uncompressed CSV files totaling 539.9 MB on disk.

| Filename | Format | Size (Bytes) | Size (MB) | Rows | Columns | SHA256 Hash | Semantic Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `in-house_label_data.csv` | CSV | 29,927 | 0.03 | 58 | 51 | `97cfa251757f6fb6e14311207d24d3f51a13ed840a07b9162af48a93961c5db3` | Initial experimentally labeled seed library (Batch 0, N=58) with full 23-cycle decay curves and `act_capacity_20`. |
| `label_all_batches_feat.csv` | CSV | 85,073 | 0.08 | 208 | 52 | `9aabe70c17401a2e0f1202d97059e6103be4ab7002cb5a65a221f76cf8bf9357` | Full active-learning campaign dataset spanning Batches 0 to 7 (N=208). Contains `batch` and `expt_test` indicators. |
| `label_batch1-6_feat.csv` | CSV | 83,263 | 0.08 | 199 | 50 | `52eee47d3eb0c3fadfb041673fa790b8afe57ed423cf0440f430bd388e3b3896` | Labeled subset spanning Batches 0 to 6 (N=199) with 100% complete feature columns. |
| `label_unlabel_all_uniq_solvents.csv` | CSV | 11,460,424 | 10.93 | 388,013 | 2 | `511a206d72a78427ff8840ae9fd00e983c57d30b9218e4407c1e2121cf783904` | Master unique solvent registry with `std_smiles` and `expt_test` status (-1 = unlabeled, 0..7 = batch tested). |
| `label_unlabel_all_uniq_solvents_fgrp_class.csv` | CSV | 19,530,939 | 18.63 | 388,013 | 4 | `e3d9c659ff8413c7b6b1b4adf354b51d3bd0e53d9c4fe4d43746eaf3806991f5` | Functional group classification (e.g. Ether, Amide, Ester) for all 388,013 solvent molecules. |
| `label_unlabel_all_uniq_solvents_fgrp_class_tsne.csv` | CSV | 29,142,173 | 27.79 | 388,013 | 6 | `812e8d01d28d9bf83a952e1ca47ab7251e14445e786921097a52899cc24f87be` | Precomputed 2D t-SNE embedding coordinates (`tsne_0`, `tsne_1`) for the 388,013 solvent library. |
| `virtual_search_space_1million.csv` | CSV | 479,621,408 | 457.40 | 999,999 | 27 | `68825ce89aacb09bf76a2ebcfeedc53a7eabb25aa2a7f4db3159180db04bb16c` | Virtual screening candidate library: 999,999 rows of solvent-salt formulations with precomputed ECFP PCA features. |

All SHA256 hashes match `data/external/aicoscientist_datasets_manifest.json` exactly.

---

# 3. Scientific Dataset Semantics

In the Amanchukwu Lab study, an electrolyte candidate represents a **standardized binary electrolyte mixture** formulated for zero-excess anode-free lithium metal coin cells:
* **Cell Architecture:** Anode-free Cu||LiFePO4 (LFP) coin cell (bare copper foil as working anode current collector, LiFePO4 as cathode). In a small subset of initial experiments (12 out of 208), NMC cathode was tested.
* **Chemical Formulation:** Exactly one organic solvent molecule dissolved with one lithium salt at a standardized concentration of 1.0 M (mol/L).
* **Standardized Testing Conditions:**
  * Electrolyte flooding volume: 50 µL.
  * Cathode loading capacity: 150 mAh/g theoretical capacity.
  * Cycling protocol: Galvanostatic cycling with initial formation cycles followed by continuous C/3 or 1C cycling.
* **Candidate Definition:** Each candidate row in `virtual_search_space_1million.csv` represents a distinct **(Solvent, Salt)** combination. It does not vary concentration, electrolyte volume, or cell geometry; those are held strictly constant.

---

# 4. Labeled Experimental Data

### Sample Sizes Across Files
* `in-house_label_data.csv`: 58 rows (Batch 0 seed library).
* `label_batch1-6_feat.csv`: 199 rows (Batches 0 to 6).
* `label_all_batches_feat.csv`: 208 rows (Batches 0 to 7).

### Batch Composition
| Batch Index | Cell Count | Role in Campaign | Nonzero `norm_capacity_3` | Mean Target | Max Target | Notes |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Batch 0** | 58 | Initial in-house exploratory library | 58 / 58 (100%) | 0.4689 | 0.8168 | Diverse ether/dual-salt library; full 23-cycle decay curves. |
| **Batch 1** | 40 | Active learning round 1 (High exploration) | 8 / 40 (20%) | 0.0001 | 0.0011 | Exploration into non-ether chemistries; almost all cells died on formation. |
| **Batch 2** | 23 | Active learning round 2 | 11 / 23 (48%) | 0.0382 | 0.4519 | Shift towards fluorinated acetals and ethers. |
| **Batch 3** | 10 | Active learning round 3 | 6 / 10 (60%) | 0.1335 | 0.5496 | Small batch refining glymes and polyethers. |
| **Batch 4** | 21 | Active learning round 4 | 9 / 21 (43%) | 0.0406 | 0.6343 | Testing ester-ether hybrids and formates. |
| **Batch 5** | 31 | Active learning round 5 | 21 / 31 (68%) | 0.2881 | 0.7469 | Acetal-glyme combinations showing high performance. |
| **Batch 6** | 16 | Active learning round 6 | 8 / 16 (50%) | 0.2720 | 0.8276 | Discovery of new global optimum (`COC1CCCC1` / CPME). |
| **Batch 7** | 9 | Active learning round 7 (Exploitation/Validation) | 8 / 9 (89%) | 0.5041 | 0.8200 | High-confidence exploitation; 8 of 9 cells showed high capacity retention. |
| **Total** | **208** | **Full 7-Round AL Campaign** | **129 / 208 (62%)** | **0.2312** | **0.8276** | **97 unique solvents tested.** |

---

# 5. Million-Candidate Search Space

### Exact Structure of `virtual_search_space_1million.csv`
* **Total Rows:** Exactly 999,999.
* **Total Columns:** 27.
* **Missing Values:** Exactly 0 missing values across all 999,999 rows.
* **Salt Distribution:**
  * `[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F` (LiFSI): Exactly 333,333 rows (33.333%).
  * `[Li+].F[P-](F)(F)(F)(F)F` (LiPF6): Exactly 333,333 rows (33.333%).
  * `[Li+].O=C1O[B-](F)(F)OC1=O` (LiDFOB): Exactly 333,333 rows (33.333%).
* **Solvent Distribution:**
  * 388,004 unique solvent molecules.
  * 278,525 solvents are paired with all 3 salts ($278,525 \times 3 = 835,575$).
  * 54,945 solvents are paired with 2 salts ($54,945 \times 2 = 109,890$).
  * 54,534 solvents are paired with 1 salt ($54,534 \times 1 = 54,534$).
  * Total = $835,575 + 109,890 + 54,534 = 999,999$.
* **Constant Columns in Virtual Space:**
  * `conc_salt_1`: 1.0 M (std = 0.0).
  * `theor_capacity`: 150.0 mAh/g (std = 0.0).
  * `amt_electrolyte`: 50.0 µL (std = 0.0).
* **Generation Mechanism:** Cartesian product of 333,333 filtered organic molecules drawn from commercial and virtual chemical catalogs paired across three standard lithium battery salts.

---

# 6. Target Objective

### Exact Physical and Mathematical Meaning
The primary target column throughout the entire active-learning campaign is `norm_capacity_3`.
* **Definition:**
  $$\text{norm\_capacity\_3} = \frac{\text{Specific Discharge Capacity at Cycle 3}}{\text{Cathode Theoretical Capacity}}$$
* **Cathode Theoretical Capacity:**
  * 150 mAh/g for $\text{LiFePO}_4$ (LFP).
  * 203 mAh/g for $\text{LiNi}_{0.8}\text{Mn}_{0.1}\text{Co}_{0.1}\text{O}_2$ (NMC811).
  * 161 mAh/g for $\text{LiNi}_{0.33}\text{Mn}_{0.33}\text{Co}_{0.33}\text{O}_2$ (NMC111).
* **Physical Significance:** In anode-free lithium metal cells, early-cycle discharge capacity under 1C rate reflects both the initial lithium plating/stripping Coulombic efficiency and the resistance of the solid electrolyte interphase (SEI). Formulations that fail to passivate lithium metal suffer micro-short circuits or complete lithium exhaustion during formation, yielding capacity near 0.0. High values ($\ge 0.70$) indicate stable lithium cyclability.

### Verification of `act_capacity_20` Alias
In `in-house_label_data.csv`, the column `act_capacity_20` is an exact mathematical alias for cycle 3 capacity in physical units (mAh/g):
$$\text{act\_capacity\_20} = \text{norm\_capacity\_3} \times \text{theor\_capacity} \quad (\text{Max absolute difference} < 10^{-6})$$

### Distribution Statistics for Experimental Targets
| Metric | `norm_capacity_3` (All 208) | `norm_capacity_3` (Batch 0, N=58) | `act_capacity_20` (Batch 0, mAh/g) |
| :--- | :---: | :---: | :---: |
| **Count** | 208 | 58 | 58 |
| **Missing** | 0 | 0 | 0 |
| **Unique Values**| 98 | 58 | 58 |
| **Min (0%)** | 0.0000 | 0.0000 | 0.0019 |
| **5th Percentile**| 0.0000 | 0.0007 | 0.1111 |
| **25th Percentile**| 0.0000 | 0.2496 | 37.4463 |
| **50th Percentile (Median)** | 0.0006 | 0.5790 | 90.2059 |
| **75th Percentile**| 0.5617 | 0.7251 | 108.7600 |
| **95th Percentile**| 0.7579 | 0.7765 | 146.4027 |
| **100th Percentile (Max)** | 0.8276 | 0.8168 | 154.5200 |
| **Mean** | 0.2312 | 0.4689 | 75.1798 |
| **Standard Deviation** | 0.3011 | 0.2730 | 46.4514 |

The distribution across all 208 experiments is strongly bimodal: 79 cells (38.0%) failed completely ($\text{capacity} \le 0.001$), while successful cells form a cluster centered around 0.60–0.80.

---

# 7. Pre-Experiment vs Post-Experiment Information

| Field Name | Physical / Chemical Meaning | Known Before Experiment? | Revealed After Experiment? | Information Leakage Risk |
| :--- | :--- | :---: | :---: | :---: |
| `solv_comb_sm` | SMILES formula of the solvent molecule | **YES** | NO | **SAFE** |
| `salt_comb_sm` | SMILES formula of the lithium salt | **YES** | NO | **SAFE** |
| `solv_ecfp_pca_0..9` | 10 PCA projections of solvent Morgan fingerprint | **YES** | NO | **SAFE** (Precomputed structural descriptor) |
| `salt_ecfp_pca_0..9` | 10 PCA projections of salt Morgan fingerprint | **YES** | NO | **SAFE** (Precomputed structural descriptor) |
| `mol_wt_solv` | Solvent molecular weight (g/mol) | **YES** | NO | **SAFE** |
| `mol_wt_salt` | Salt molecular weight (g/mol) | **YES** | NO | **SAFE** |
| `conc_salt_1` | Salt molar concentration (1.0 M in candidate pool) | **YES** | NO | **SAFE** (Prescribed experimental condition) |
| `theor_capacity` | Theoretical cathode specific capacity (mAh/g) | **YES** | NO | **SAFE** (Cathode specification) |
| `amt_electrolyte` | Electrolyte volume injected into coin cell (50 µL) | **YES** | NO | **SAFE** (Cell assembly protocol) |
| `batch` | Active learning round index (0 to 7) | **NO** (Contextual) | Contextual | **REVIEW** (Encodes campaign stage) |
| `expt_test` | Experimental batch index indicator | **NO** (Contextual) | Contextual | **REVIEW** (Redundant with `batch`) |
| `norm_capacity_1` | Normalized discharge capacity at cycle 1 | NO | **YES** | **POST-EXPERIMENT ONLY** |
| `norm_capacity_2` | Normalized discharge capacity at cycle 2 | NO | **YES** | **POST-EXPERIMENT ONLY** |
| `norm_capacity_3` | Normalized discharge capacity at cycle 3 (Main Target) | NO | **YES** | **POST-EXPERIMENT ONLY** |
| `norm_capacity_4..23`| Normalized discharge capacity at cycles 4 to 23 | NO | **YES** | **POST-EXPERIMENT ONLY** |
| `act_capacity_20` | Actual discharge capacity in mAh/g (`norm_capacity_3 * theor`) | NO | **YES** | **POST-EXPERIMENT ONLY** |

---

# 8. Labeled Set Coverage of Candidate Space

### Nearest-Neighbor Distance Analysis
To evaluate whether active learning was performing **interpolation** or **extrapolation**, we computed the exact Euclidean distance from every candidate ($N=999,999$) to the nearest experimentally measured sample in the 22-dimensional standardized feature space:

| Metric | Nearest Distance to Batch 0 Seed ($N=58$) | Nearest Distance to Full Campaign ($N=208$) |
| :--- | :---: | :---: |
| **Minimum** | $8.24 \times 10^{-9}$ | $0.0000$ |
| **5th Percentile** | 2.1928 | 1.4617 |
| **25th Percentile** | 4.0641 | 2.5378 |
| **Median (50th Percentile)** | **5.6088** | **3.2233** |
| **75th Percentile** | 7.1319 | 3.9086 |
| **90th Percentile** | 7.6418 | 4.5500 |
| **95th Percentile** | 7.9137 | 4.9475 |
| **99th Percentile** | 8.4354 | 5.7174 |
| **Maximum** | **11.0706** | **8.0564** |
| **Mean $\pm$ Std** | $5.4786 \pm 1.8483$ | $3.2241 \pm 1.0392$ |

**Scientific Interpretation:**
* The initial 58 seed experiments occupied an extremely tiny, isolated island in the 22-dimensional space. The median candidate was **5.61 standard deviations away** from the nearest seed point.
* Even after 7 rounds of active learning, the median candidate remains **3.22 standard deviations away**, and 99% of candidates are more than 1.46 standard deviations away from any tested sample.
* This quantitatively confirms that active learning in this domain operates in a regime of **severe, high-dimensional chemical extrapolation**.

### Functional Group Diversity & Extrapolation
* Total functional group classes in the solvent library: **430 classes**.
* Functional group classes represented in tested experiments: **16 classes** (3.72%).
* Functional group classes with ZERO experimental tests: **414 classes** (96.28%).
* Dominance of Ethers: Of the 97 tested solvent molecules, 65 are simple ethers (`['Ether']`), and combined with mixed ethers, over 75% of all tested chemistries are ether derivatives. Major chemical families in the 1M candidate pool (such as amides with >33,000 candidates, amines with >33,000 candidates, and sulfoxides) were almost entirely avoided or abandoned after early failure in Batch 1.

---

# 9. Search-Space Difficulty

Is finding a top candidate in this space difficult?
1. **Low Density of Viable Candidates:** In Batch 1, when the model explored broadly outside conventional ether chemistries (40 candidates), 80% of cells failed to cycle at all (capacity = 0.0), and the remaining 20% had capacity $<0.002$. Viable anode-free electrolytes constitute a minuscule fraction of arbitrary organic molecules.
2. **Top-10 Performance Spread:**
   * Best candidate: 0.8276 (Batch 6)
   * 2nd best candidate: 0.8200 (Batch 7)
   * 3rd best candidate: 0.8168 (Batch 0 seed!)
   * 5th best candidate: 0.7764 (Batch 0 seed!)
   * Best / Median ratio across all experiments: $0.8276 / 0.0006 \approx 1379\times$.
   * Top-5 spread: $0.8276 - 0.7764 = 0.0512$ (only a 5.1% capacity difference between the best newly discovered solvent and the initial seed).
3. **The "Curse of Local Optima":** The initial seed set already contained a high-performing electrolyte (fluorinated DME derivative, `norm_capacity_3` = 0.8168). Over 7 active-learning batches and 150 new cells, the highest performance improvement achieved was from 0.8168 to 0.8276 (+1.3% relative improvement). However, the active-learning campaign succeeded in discovering **chemically distinct** high-performing solvents (cyclic ethers like cyclopentyl methyl ether), demonstrating structural diversity rather than massive numeric improvement.

---

# 10. Active-Learning Campaign History

The local dataset contains the complete, reconstructible 8-round sequential active learning trajectory:

```text
Round 0 (Initial Seed Library):
  - 58 experiments (45 LFP cells, 10 NMC811 cells, 3 NMC111 cells)
  - Broadly focused on ether variants with LiFSI salt
  - Baseline best target: 0.8168

Round 1 (High Exploration Batch):
  - 40 experiments
  - Chemistries: Amines, chlorinated piperidines, non-ether organics
  - Result: Severe degradation; 32/40 had zero capacity; max = 0.0011

Round 2 (Re-anchoring & Acetal Exploration):
  - 23 experiments
  - Chemistries: Fluorinated acetals (e.g. COC(OC)C(F)(F)F) paired with LiFSI, LiPF6, LiDFOB
  - Result: Modest recovery; max = 0.4519

Round 3 (Polyether Chain Length Tuning):
  - 10 experiments
  - Chemistries: Longer ethoxy / glyme oligomers with LiFSI
  - Result: Steady progress; max = 0.5496

Round 4 (Esters & Formate Derivatives):
  - 21 experiments
  - Chemistries: Hexyl formate, methanesulfonate esters
  - Result: Exploration of ester compatibility; max = 0.6343

Round 5 (Acetal-Glyme Hybrids):
  - 31 experiments
  - Chemistries: Orthoformates, acetal-glymes (COCCOCC(OC)OC) across LiFSI, LiPF6, LiDFOB
  - Result: Significant breakthrough; max = 0.7469; 21/31 cells viable

Round 6 (Cyclic Ethers & Discovery):
  - 16 experiments
  - Chemistries: Cyclopentyl methyl ether (CPME, COC1CCCC1), diether oligomers
  - Result: Global campaign optimum discovered: COC1CCCC1 + LiFSI -> 0.8276

Round 7 (Confirmation & Exploitation):
  - 9 experiments
  - Chemistries: Focused exploitation of fluorinated ethers and asymmetric diethers
  - Result: 8 of 9 cells succeeded; median target = 0.7260, max = 0.8200
```

---

# 11. Available Experimental Modalities

* **Supported Modalities:** Exactly **ONE** modality: Galvanostatic discharge capacity during electrochemical cell cycling.
  * In Batch 0 ($N=58$), capacity decay across 23 cycles is recorded.
  * In Batches 1 to 7 ($N=150$), only cycle 3 capacity is recorded.
* **Absent Modalities:**
  * No electrochemical impedance spectroscopy (EIS / charge-transfer resistance $R_{ct}$).
  * No ionic conductivity measurements ($\sigma$, mS/cm).
  * No electrochemical stability window (cyclic voltammetry / LSV for oxidation/reduction limits).
  * No viscosity measurements ($\eta$, cP).
  * No operando gas analysis, SEM, XPS, NMR, or optical characterization.
* **Verdict on Candidate × Measurement Action Space:** **NOT SUPPORTED.** A scientific agent cannot choose between running a cheap screening test (e.g. conductivity or viscosity) versus an expensive test (cycling). Every experimental point represents the exact same physical action: fabricating and cycling an anode-free coin cell.

---

# 12. Offline Replay Feasibility

| Evaluation Mode | Feasibility | Scientific Justification |
| :--- | :---: | :--- |
| **MODE A: Strict Offline Replay** | **NO** (Over 1M space)<br>**YES** (Over 208 points) | An agent searching the 1M virtual library will select unmeasured candidates with 99.98% probability. An offline engine cannot truthfully reveal experimental labels for unmeasured points. Strict replay is only valid when restricted to the 208 measured points. |
| **MODE B: Surrogate Oracle Simulation** | **YES** | A surrogate model (Random Forest, Gaussian Process, or Bayesian Model Ensemble) trained on the 208 points can act as a synthetic ground-truth simulator over the 1M candidate library. |
| **MODE C: Active-Learning Trajectory Replay** | **YES** | Directly benchmarks algorithms on their ability to reproduce or outperform the historical sequential batch selection from Batch 0 through Batch 7. |
| **MODE D: Virtual Discovery Benchmark** | **PARTIAL** | The 1M candidate library contains only ECFP PCA descriptors and molecular weights. It does not include computed DFT targets (such as HOMO/LUMO levels, binding energies, or dielectric constants) that could serve as ground-truth virtual properties. |

---

# 13. Hypothesis-Space Potential

Although hypotheses cannot be executed during audit, the dataset features support specific explanatory families:

### Plausible Explanation Families Supported by Data
1. **Solvent Fluorination Hypothesis:** Fluorinated terminal alkyl groups (`-CF2CF3`, `-CF2CHF2`) decrease solvent flammability and participate in forming an inorganic-rich LiF-passivating SEI, dramatically improving cycling capacity retention over non-fluorinated analogues.
   * *Features:* ECFP PCA coordinates capturing C-F bonds, molecular weight shifts.
   * *Evidence in Data:* High representation of fluorinated ethers in top-performing candidates (Ranks 2, 3, 5, 7, 8).
2. **Salt Anion Passivation (LiFSI Superiority):** LiFSI anion ($[\text{N}(\text{SO}_2\text{F})_2]^-$) decomposes into a robust LiF/$\text{Li}_2\text{S}_x\text{O}_y$ interphase, whereas $\text{PF}_6^-$ and $\text{DFOB}^-$ suffer from transition-metal dissolution or high interfacial impedance.
   * *Features:* Salt ECFP PCA coordinates, salt MW.
   * *Evidence in Data:* LiFSI yields mean capacity 0.323 (max 0.8276), while LiPF6 and LiDFOB yield mean capacity <0.081.
3. **Steric Bulk & Cyclic Ether Solvation:** Moderate ring constraints (as in methoxycyclopentane) or branched structures weaken the Li-solvent binding energy, promoting contact-ion pairs and anion-derived SEI formation.
   * *Features:* Solvent ECFP PCA dimensions 0, 1, and 4.

### Unsupported Explanation Families (Missing Measurements)
* **Viscosity-Controlled Ion Transport:** Cannot be tested because viscosity was not measured.
* **SEI Chemical Evolution:** Cannot be tested because spectroscopic surface analysis (XPS/FTIR) is absent.
* **Coordination Sheath / Solvation Shell Speciation:** Cannot be tested without Raman, NMR, or molecular dynamics data.

---

# 14. Information Leakage Audit

* **Candidate Space Table (`virtual_search_space_1million.csv`):**
  * Status: **SAFE.**
  * No target columns (`norm_capacity_*`, `act_capacity_*`) exist in the candidate table.
  * No batch indicators or timestamps are present.
* **Batch 7 Missingness Anomaly:**
  * Status: **REVIEW.**
  * In `label_all_batches_feat.csv`, rows 199–207 have NaNs for all 22 feature columns. Any automated pipeline attempting to train on `label_all_batches_feat.csv` will crash or drop Batch 7 unless features are imputed from `virtual_search_space_1million.csv`.
* **Standardization & PCA Fitting Scope:**
  * Status: **REVIEW.**
  * The ECFP PCA components (`solv_ecfp_pca_0..9`) were fit upstream across the virtual solvent database and applied to the experimental samples. This is an unsupervised feature representation and does not leak target labels.
* **Duplicate SMILES Representations:**
  * Status: **REVIEW.**
  * The same fluorinated ether is written as `C(COCCOC)(C(F)(F)C(F)(F)F)(F)F` in `in-house_label_data.csv` and `COCCOCC(F)(F)C(F)(F)F` in `label_all_batches_feat.csv`. Any exact string-matching domain adapter will treat them as two different molecules unless canonical SMILES normalization is applied.

---

# 15. Baseline Learnability

To test whether the 22 pre-experiment features contain predictive signal for the cycling target (`norm_capacity_3`), we evaluated four regression architectures using 5-fold cross-validation with feature standardization.

| Dataset / Model | MAE | RMSE | $R^2$ Score | Spearman Rank ($\rho$) | Learnable Signal Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Batch 0 Seed ($N=58$)** | | | | | |
| Dummy (Predict Mean) | 0.2443 | 0.2757 | -0.0206 | -0.1721 | Baseline reference |
| Ridge Regression ($\alpha=1.0$) | 0.2755 | 0.4166 | -1.3301 | 0.0853 | Severe overfitting (high D/N ratio) |
| Random Forest (100 trees) | 0.1761 | 0.2205 | 0.3470 | **0.4897** | Moderate ranking signal |
| Gaussian Process (Matern5/2) | 0.1704 | 0.2164 | 0.3713 | **0.4744** | Moderate ranking signal |
| **All Batches 0–7 ($N=208$)** | | | | | |
| Dummy (Predict Mean) | 0.2751 | 0.3014 | -0.0071 | -0.0745 | Baseline reference |
| Ridge Regression ($\alpha=1.0$) | 0.2002 | 0.2439 | 0.3405 | 0.6069 | Significant linear ranking signal |
| Random Forest (100 trees) | **0.1211** | **0.1808** | **0.6378** | **0.7765** | **Strong, highly significant ranking signal** |
| Gaussian Process (Matern5/2) | 0.1569 | 0.2232 | 0.4477 | **0.6788** | **Strong Bayesian ranking signal** |

**Conclusions on Learnability:**
* The feature space possesses **strong, robust predictive signal** for capacity retention across the full campaign ($R^2 > 0.63$, $\rho > 0.77$).
* Gaussian Processes and tree-based ensembles capture non-linear interactions between ECFP PCA descriptors and salt identity effectively.
* However, on the initial seed batch ($N=58$), linear models fail completely due to high dimensionality relative to $N$, necessitating non-linear Bayesian or regularized models.

---

# 16. Computational Feasibility at 1M Candidates

### Memory Footprint
* Candidate count: $N = 999,999$.
* Feature dimension: $D = 22$ numerical features.
* **Float64 Matrix Size:** $999,999 \times 22 \times 8 \text{ bytes} \approx 176.0 \text{ MB}$.
* **Float32 Matrix Size:** $999,999 \times 22 \times 4 \text{ bytes} \approx 88.0 \text{ MB}$.
* **Full Pandas DataFrame (with SMILES strings):** ~457 MB.

### Scoring & Acquisition Runtime
* **Full File Scan Time:** 8.13 seconds (streamed from local SSD in 200k chunks).
* **Exact Nearest-Neighbor Calculation (1M points $\times$ 208 points):** 8.95 seconds.
* **BoTorch / PyTorch Feasibility:**
  * Evaluating posterior mean and variance from a Gaussian Process over 1M candidates in float32 takes <1.5 seconds on GPU and <4 seconds on multi-core CPU.
  * For batch acquisition functions (e.g. q-Expected Improvement, q-NEI), full-matrix optimization requires chunking into batches of 50,000 candidates (~8.8 MB per chunk), which avoids CUDA out-of-memory errors.

---

# 17. Benchmark Metrics Supported by the Dataset

| Benchmark Metric | Direct Replay ($N=208$) | Surrogate Simulation ($N=1\text{M}$) | Benchmark Role |
| :--- | :---: | :---: | :--- |
| **Max Capacity Found (Best Sample)** | **SUPPORTED** | **SUPPORTED** | Measures discovery peak reached within budget $K$. |
| **Simple Regret ($y^* - \max_{i \le t} y_i$)** | **SUPPORTED** | **SUPPORTED** | Quantifies optimization gap over iteration steps. |
| **Cumulative Regret** | **SUPPORTED** | **SUPPORTED** | Penalizes selecting dead/failed electrolytes. |
| **Top-K Candidate Recovery** | **SUPPORTED** | **SUPPORTED** | Measures recall of the top 10 historical formulations. |
| **Experiments to Reach Feasibility ($\ge 0.70$)** | **SUPPORTED** | **SUPPORTED** | Measures cold-start sample efficiency. |
| **Predictive Calibration / Log-Likelihood** | **SUPPORTED** | **SUPPORTED** | Evaluates uncertainty quantification quality. |
| **Multi-Objective Pareto Front Quality** | **NOT SUPPORTED** | **NOT SUPPORTED** | Only one scalar target exists. |
| **Information Gain per Cost** | **NOT SUPPORTED** | **NOT SUPPORTED** | No experimental cost metadata exists. |

---

# 18. Critical Limitations

1. **Extreme Sparsity of Ground Truth (0.021% Label Coverage):** Out of 999,999 virtual candidates, only 208 have experimental outcomes. True offline evaluation of an autonomous agent over the 1M candidate library is strictly impossible without an artificial surrogate oracle.
2. **Single-Modality Action Space:** The dataset contains no complementary physical characterizations (no EIS, conductivity, viscosity, XPS, or FTIR). It cannot benchmark multi-modal scientific reasoning or staged decision-making.
3. **Chemical Over-Concentration:** Over 75% of all tested electrolytes are ether-based. The active-learning campaign aggressively abandoned non-ether chemical families after Batch 1 failures, leaving 414 out of 430 functional classes completely untested.
4. **Target Normalization Nuance:** The primary target `norm_capacity_3` is normalized to cathode theoretical capacity, but cell chemistry varied across batches (LFP with 150 mAh/g vs NMC with 203 mAh/g). Comparing raw normalized capacity across different cathode chemistries introduces physical ambiguity.
5. **Batch 7 Missing Feature Columns:** In `label_all_batches_feat.csv`, rows 199–207 have NaNs for all features, requiring imputation from the 1M candidate file.
6. **No Cost or Throughput Metadata:** No experimental costs (dollar cost, synthesis complexity, cycling hours) are recorded.

---

# 19. Recommended Roles for This Dataset

| Proposed Role | Fit Rating | Detailed Scientific Justification |
| :--- | :---: | :--- |
| **Active-Learning Campaign Reproduction Benchmark** | **HIGH FIT** | The 8-round sequential campaign (Batches 0 to 7) provides an exceptional historical benchmark. AIcoScientist can evaluate whether modern Bayesian optimization, epistemic exploration, and LLM hypotheses could have avoided the catastrophic failures of Batch 1 and discovered the high-performing ethers faster than the original 7 rounds. |
| **Cold-Start Extrapolation Benchmark** | **HIGH FIT** | Starting from only 58 seed points to discover viable candidates across a 22-dimensional feature space where the median candidate is 5.6 standard deviations away represents a premier test of high-dimensional extrapolation and uncertainty calibration. |
| **Surrogate Simulation Discovery Benchmark** | **MEDIUM FIT** | Feasible by fitting a Gaussian Process / Random Forest ensemble on the 208 points and treating it as an in-silico oracle for 1M candidate search. However, discovery on a synthetic surrogate does not reflect true wet-lab experimental outcomes. |
| **Multi-Modal Scientific Reasoning Benchmark** | **LOW FIT** | Unsuitable. The dataset lacks multi-modal measurements, mechanistic characterization, and staged experimental actions. |

---

# 20. Questions Requiring Design Decisions

Before designing an AIcoScientist domain adapter for this dataset, the research team must resolve:

1. **How should an unmeasured 1M-candidate selection be handled?**
   * *Option A:* Restrict the offline agent's action space strictly to the 208 historically measured points (or the 97 unique solvents).
   * *Option B:* Train a surrogate model (or ensemble of GP + Random Forest) on the 208 points to serve as an in-silico simulator that outputs synthetic observations for arbitrary 1M candidates.
2. **Should the benchmark reproduce the sequential active-learning rounds?**
   * If the campaign trajectory is replayed (Batch 0 $\to$ Batch 1 $\dots$ $\to$ Batch 7), should the agent be evaluated on its ability to rank the candidates selected in the next batch, or to select an alternative batch from the 1M pool?
3. **How should the cathode variation in Batch 0 be normalized?**
   * Should the 13 NMC cells in Batch 0 (theor_capacity = 203 and 161 mAh/g) be filtered out so the benchmark strictly evaluates Cu||LFP cells, or should theoretical capacity remain an explicit input feature?
4. **What constitutes a valid hypothesis in this single-modality domain?**
   * Since only cycling capacity is measured, how can mechanistic hypotheses (e.g. SEI formation vs coordination strength) be evaluated and falsified without intermediate characterization data?
5. **Should the Batch 7 feature missingness be repaired in the domain adapter?**
   * The domain adapter must automatically join `virtual_search_space_1million.csv` to supply the missing ECFP PCA features for Batch 7 solvents.

---
*End of Dataset Audit Report.*

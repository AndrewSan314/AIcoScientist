# Scientific Dataset Audit Report: Amanchukwu Lab Anode-Free Electrolyte Search Space (Corrected)

**Dataset Identifier:** `AmanchukwuLab/AL-anode-free` (`al_anode_free_2025`)  
**Associated Publication:** *Active learning accelerates electrolyte solvent screening for anode-free lithium metal batteries*, Nature Communications (2025), DOI: [10.1038/s41467-025-63303-7](https://doi.org/10.1038/s41467-025-63303-7)  
**Authors:** Peiyuan Ma, Ritesh Kumar, Ke-Hsin Wang, Chibueze V. Amanchukwu (Pritzker School of Molecular Engineering, University of Chicago)  
**Local Dataset Path:** `data/external/al_anode_free_2025/`  
**Audit Revision:** Correction Batch (September 2026)  
**Auditor:** AIcoScientist Data Architecture & Verification Engine  

---

# 1. Executive Summary

### Answers to Mandatory Core Questions:
1. **Raw 1M candidate count:** Exactly **999,999** formulation rows.
2. **Number of unique solvents:** **388,004** in the candidate library (and 388,013 in the master catalog); exactly **97** unique solvents were experimentally tested.
3. **Raw labeled ML-row count:** **208** rows in the aggregated training table (`label_all_batches_feat.csv`).
4. **Independent wet-lab experiment count:** **UNKNOWN**.
5. **Why independent experiment count is unknown:** The aggregated table contains machine learning training representations where **115 rows (across 39 solvent-batch groups) have identical target values copied across 3 different salts**. Without physical cell serials, run logs, or lab timestamps, it is impossible to distinguish whether cells were fabricated 3 times independently or tested once and pseudo-expanded across salts for model training.
6. **Exact scientific target:** **$C_{\text{norm}}^{20}$** (Normalized discharge capacity at the **20th cycle**, defined as $C_{\text{dis}}^{20} / C_{\text{theor}}$).
7. **What raw field `norm_capacity_3` actually represents:** It is a historical dataset column name / variable encoding for **$C_{\text{norm}}^{20}$**, verified numerically against `act_capacity_20 / theor_capacity` with maximum absolute error $< 3.34 \times 10^{-10}$ across all Batch 0 samples (zero exceptions). It MUST NOT be interpreted literally as cycle 3.
8. **Labeled rows matching the 1M candidate contract:** **139 rows** under raw string matching, **142 rows** under canonical salt alias matching, and **151 rows** when Batch-7 cell parameters are recovered from the candidate pool.
9. **Distinct compatible solvents:** **63 solvents** (raw), **66 solvents** (canonical), and **75 solvents** (recovered).
10. **Full 1M wet-lab replay feasibility:** **NO.** Only ~0.02% of the candidate library has historical experimental records.
11. **Historical trajectory reconstruction feasibility:** **YES.** The 8 sequential campaign rounds (Batches 0 to 7) are fully documented and reconstructible.
12. **Counterfactual closed-loop experimental replay feasibility:** **NO.** If an agent chooses an unmeasured formulation, no true experimental ground truth exists.
13. **Multimodal candidate × measurement action space support:** **NOT SUPPORTED.** The dataset contains only a single physical experimental modality: galvanostatic coin-cell cycling discharge capacity.
14. **Biggest scientifically valid benchmark opportunity:** **Historical active-learning campaign reconstruction and cold-start extrapolation benchmarking** (testing whether BO / epistemic policies can discover high-performing ethers faster and avoid Batch-1 catastrophic exploration failures).
15. **Biggest limitation:** **Severe label sparsity (99.98% unmeasured)** combined with **target-copy expansion across salts** in the ML modeling table and complete absence of complementary physical characterization measurements.

---

# 2. File Inventory

The local dataset directory `data/external/al_anode_free_2025/` contains 7 uncompressed CSV files totaling 539.9 MB on disk.

| Filename | Format | Size (Bytes) | Size (MB) | Rows | Columns | SHA256 Hash | Semantic Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `in-house_label_data.csv` | CSV | 29,927 | 0.03 | 58 | 51 | `97cfa251757f6fb6e14311207d24d3f51a13ed840a07b9162af48a93961c5db3` | Initial experimentally labeled seed library (Batch 0, N=58) with full 23-cycle decay curves and `act_capacity_20`. |
| `label_all_batches_feat.csv` | CSV | 85,073 | 0.08 | 208 | 52 | `9aabe70c17401a2e0f1202d97059e6103be4ab7002cb5a65a221f76cf8bf9357` | Full active-learning campaign ML representation dataset spanning Batches 0 to 7 (N=208). Contains target-copied rows. |
| `label_batch1-6_feat.csv` | CSV | 83,263 | 0.08 | 199 | 50 | `52eee47d3eb0c3fadfb041673fa790b8afe57ed423cf0440f430bd388e3b3896` | Labeled subset spanning Batches 0 to 6 (N=199) with complete feature columns. |
| `label_unlabel_all_uniq_solvents.csv` | CSV | 11,460,424 | 10.93 | 388,013 | 2 | `511a206d72a78427ff8840ae9fd00e983c57d30b9218e4407c1e2121cf783904` | Master unique solvent registry with `std_smiles` and `expt_test` status (-1 = unlabeled, 0..7 = batch tested). |
| `label_unlabel_all_uniq_solvents_fgrp_class.csv` | CSV | 19,530,939 | 18.63 | 388,013 | 4 | `e3d9c659ff8413c7b6b1b4adf354b51d3bd0e53d9c4fe4d43746eaf3806991f5` | Functional group classification (e.g. Ether, Amide, Ester) for all 388,013 solvent molecules. |
| `label_unlabel_all_uniq_solvents_fgrp_class_tsne.csv` | CSV | 29,142,173 | 27.79 | 388,013 | 6 | `812e8d01d28d9bf83a952e1ca47ab7251e14445e786921097a52899cc24f87be` | Precomputed 2D t-SNE embedding coordinates (`tsne_0`, `tsne_1`) for the 388,013 solvent library. |
| `virtual_search_space_1million.csv` | CSV | 479,621,408 | 457.40 | 999,999 | 27 | `68825ce89aacb09bf76a2ebcfeedc53a7eabb25aa2a7f4db3159180db04bb16c` | Virtual screening candidate library: 999,999 rows of solvent-salt formulations with precomputed ECFP PCA features. |

All SHA256 hashes match `data/external/aicoscientist_datasets_manifest.json` exactly.

---

# 3. Scientific Dataset Semantics

In the Amanchukwu Lab study, an electrolyte candidate represents a **standardized binary electrolyte mixture** formulated for zero-excess anode-free lithium metal coin cells:
* **Cell Architecture:** Anode-free Cu||LiFePO4 (LFP) coin cell (bare copper foil as working anode current collector, LiFePO4 as cathode). In a small subset of initial experiments (15 out of 208), NMC cathode variants were tested.
* **Chemical Formulation:** Exactly one organic solvent molecule dissolved with one lithium salt at a standardized concentration of 1.0 M (mol/L).
* **Standardized Testing Conditions:**
  * Electrolyte flooding volume: 50 µL.
  * Cathode loading capacity: 150 mAh/g theoretical capacity.
  * Cycling protocol: Galvanostatic cycling with initial formation cycles followed by continuous cycling at 1C rate.
* **Candidate Definition:** Each candidate row in `virtual_search_space_1million.csv` represents a distinct **(Solvent, Salt)** combination under standardized testing conditions.

---

# 4. Labeled Experimental Data

### Sample Sizes Across Files
* `in-house_label_data.csv`: 58 rows (Batch 0 seed library).
* `label_batch1-6_feat.csv`: 199 rows (Batches 0 to 6).
* `label_all_batches_feat.csv`: 208 rows (Batches 0 to 7 ML representation).

### ML Training Rows vs. Physical Experiments Taxonomy
A critical finding of this audit is that **training rows $\ne$ independent physical experiments**:
* `raw_labeled_training_rows`: **208**.
* `unique_solvents`: **97**.
* `unique_salts`: **6**.
* `unique_solvent_salt_pairs`: **175**.
* `unique_full_condition_rows`: **190**.
* `target_repeated_across_salts_groups`: **39 groups** (comprising **115 rows**).
* `independent_wet_lab_records_estimate`: **UNKNOWN** (due to target copy expansion across salts without physical serials).

### Batch Structure
| Batch Index | Cell Count | Role in Campaign | Nonzero $C_{\text{norm}}^{20}$ | Mean Target | Max Target | Notes |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Batch 0** | 58 | Initial in-house exploratory library | 58 / 58 (100%) | 0.4689 | 0.8168 | Diverse ether/dual-salt library; full 23-cycle decay curves. |
| **Batch 1** | 40 | Active learning round 1 (High exploration) | 8 / 40 (20%) | 0.0001 | 0.0011 | Exploration into non-ether chemistries; almost all cells died on formation. |
| **Batch 2** | 23 | Active learning round 2 | 11 / 23 (48%) | 0.0382 | 0.4519 | Shift towards fluorinated acetals and ethers. |
| **Batch 3** | 10 | Active learning round 3 | 6 / 10 (60%) | 0.1335 | 0.5496 | Small batch refining glymes and polyethers. |
| **Batch 4** | 21 | Active learning round 4 | 9 / 21 (43%) | 0.0406 | 0.6343 | Testing ester-ether hybrids and formates. |
| **Batch 5** | 31 | Active learning round 5 | 21 / 31 (68%) | 0.2881 | 0.7469 | Acetal-glyme combinations showing high performance. |
| **Batch 6** | 16 | Active learning round 6 | 8 / 16 (50%) | 0.2720 | 0.8276 | Discovery of new campaign optimum (`COC1CCCC1` / CPME). |
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
* **Solvent Distribution & Pairing Semantics:**
  * Exactly **388,004 unique solvent molecules**.
  * **278,525 solvents** are paired with all 3 salts ($278,525 \times 3 = 835,575$).
  * **54,945 solvents** are paired with 2 salts ($54,945 \times 2 = 109,890$).
  * **54,534 solvents** are paired with 1 salt ($54,534 \times 1 = 54,534$).
  * Total = $835,575 + 109,890 + 54,534 = 999,999$.
  * *Notice:* This is **not a full Cartesian product** of 333,333 solvents $\times$ 3 salts, but rather an unbalanced allocation totaling exactly 333,333 rows per salt.
* **Effective Search Space & Vector Uniqueness:**
  * Unique 22D continuous feature vectors: **999,326**.
  * Duplicate 22D feature vectors: **673**.
  * *Cause of Duplicate Vectors:* Formatting variants of the same chemical (e.g. atom-mapped SMILES vs non-atom-mapped SMILES such as `C[N:5]1[CH2:4][CH2:3][N:2]([CH3:1])[C:7](=[O:8])[CH2:6]1` vs `CN1CCN(C)C(=O)C1`) that share identical ECFP PCA descriptors and MW.
* **Constant Protocol Columns:**
  * `conc_salt_1`: 1.0 M (std = 0.0).
  * `theor_capacity`: 150.0 mAh/g (std = 0.0).
  * `amt_electrolyte`: 50.0 µL (std = 0.0).

---

# 6. Target Objective

### Corrected Scientific Semantics: $C_{\text{norm}}^{20}$
* **Scientific Target:** **$C_{\text{norm}}^{20}$**
* **Physical Meaning:** Normalized specific discharge capacity at the **20th cycle** relative to theoretical cathode capacity:
  $$C_{\text{norm}}^{20} = \frac{\text{Specific Discharge Capacity at Cycle 20 (mAh/g)}}{\text{Theoretical Cathode Capacity (mAh/g)}}$$
* **Reconciliation with Code:** In the author's codebase, this objective is assigned to column `norm_capacity_3`. This is a historical variable name / column encoding and MUST NOT be described as cycle 3.
* **Numerical Alias Validation:** In Batch 0 (`in-house_label_data.csv`), the physical measurement `act_capacity_20` satisfies:
  $$\frac{\text{act\_capacity\_20}}{\text{theor\_capacity}} = \text{norm\_capacity\_3}$$
  with **maximum absolute error $= 3.33 \times 10^{-10}$** across all 58 rows (zero exceptions).

### Distribution Statistics for $C_{\text{norm}}^{20}$
| Metric | $C_{\text{norm}}^{20}$ (All 208 Rows) | $C_{\text{norm}}^{20}$ (Batch 0 Seed, N=58) | `act_capacity_20` (Batch 0, mAh/g) |
| :--- | :---: | :---: | :---: |
| **Count** | 208 | 58 | 58 |
| **Min (0%)** | 0.0000 | 0.0000 | 0.0019 |
| **5th Percentile**| 0.0000 | 0.0007 | 0.1111 |
| **25th Percentile**| 0.0000 | 0.2496 | 37.4463 |
| **50th Percentile (Median)** | 0.0006 | 0.5790 | 90.2059 |
| **75th Percentile**| 0.5617 | 0.7251 | 108.7600 |
| **95th Percentile**| 0.7579 | 0.7765 | 146.4027 |
| **100th Percentile (Max)** | 0.8276 | 0.8168 | 154.5200 |
| **Mean $\pm$ Std** | $0.2312 \pm 0.3011$ | $0.4689 \pm 0.2730$ | $75.1798 \pm 46.4514$ |

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
| `theor_capacity` | Theoretical cathode specific capacity (150 mAh/g) | **YES** | NO | **SAFE** (Cathode specification) |
| `amt_electrolyte` | Electrolyte volume injected into coin cell (50 µL) | **YES** | NO | **SAFE** (Cell assembly protocol) |
| `batch` | Active learning round index (0 to 7) | **NO** (Contextual) | Contextual | **REVIEW** (Encodes campaign stage) |
| `expt_test` | Experimental batch index indicator | **NO** (Contextual) | Contextual | **REVIEW** (Redundant with `batch`) |
| `norm_capacity_3` | Raw column encoding $C_{\text{norm}}^{20}$ (Main Target) | NO | **YES** | **POST-EXPERIMENT ONLY** |
| `norm_capacity_1..23`| Multi-cycle decay profile (available in Batch 0) | NO | **YES** | **POST-EXPERIMENT ONLY** |
| `act_capacity_20` | Physical specific capacity in mAh/g ($C_{\text{norm}}^{20} \times \text{theor}$) | NO | **YES** | **POST-EXPERIMENT ONLY** |

---

# 8. Labeled Set Coverage of Candidate Space

### Domain-Matched Nearest-Neighbor Coverage Analysis
To prevent domain mixing between non-standard training rows and the standardized 1M candidate contract, we report three separate coverage metrics:
* **Coverage A (Historical Seed, $N=58$):** Candidate space $\to$ raw Batch 0 seed rows.
* **Coverage B (Full Training Representation, $N=208$):** Candidate space $\to$ all 208 ML rows (with Batch-7 recovered features).
* **Coverage C (Virtual-Pool-Compatible Subset, $N=151$, PRIMARY):** Candidate space $\to$ only rows matching the 1M candidate pool protocol contract (1.0 M conc, 150 mAh/g theor capacity, 50 µL vol, candidate-pool salts).

| Metric | Coverage A: Seed ($N=58$) | Coverage B: Full ML ($N=208$) | Coverage C: Pool-Compatible ($N=151$, PRIMARY) |
| :--- | :---: | :---: | :---: |
| **Minimum** | $8.24 \times 10^{-9}$ | 0.0000 | 0.0000 |
| **5th Percentile** | 2.1928 | 1.4617 | 1.5036 |
| **25th Percentile** | 4.0641 | 2.5378 | 2.6108 |
| **Median (50th Percentile)** | **5.6088** | **3.2233** | **3.3129** |
| **75th Percentile** | 7.1319 | 3.9086 | 4.0152 |
| **90th Percentile** | 7.6418 | 4.5500 | 4.6710 |
| **95th Percentile** | 7.9137 | 4.9475 | 5.0921 |
| **99th Percentile** | 8.4354 | 5.7174 | 5.8441 |
| **Maximum** | **11.0706** | **8.0564** | **8.0564** |
| **Mean $\pm$ Std** | $5.4786 \pm 1.8483$ | $3.2241 \pm 1.0392$ | $3.3325 \pm 1.0617$ |

**Scientific Interpretation:**
In the primary domain-matched metric (Coverage C), the median candidate in the 1M library is **3.31 standard deviations away** from the nearest pool-compatible measured formulation. The active-learning campaign operated in a regime of **severe high-dimensional chemical extrapolation**.

### Functional Group Diversity & Extrapolation
* Total functional classes in master solvent library: **430 classes**.
* Functional classes with $\ge 1$ experiment: **16 classes** (3.72%).
* Functional classes with ZERO experiments: **414 classes** (96.28%).
* Solvent-Catalog Random Subsampling Diversity Risk: Naive random subsampling of the solvent catalog drops rare chemical families rapidly: a 1k sample loses 62.8% of classes; a 10k sample loses 32.1%; a 100k sample loses 10.7%.

---

# 9. Search-Space Difficulty

1. **Severe Density of Inactive Formulations:** When active learning explored outside ethers in Batch 1 (40 formulations), 80% failed to cycle (capacity $= 0.0$), and the rest had capacity $<0.002$. Viable anode-free electrolytes represent an extremely sparse subset of arbitrary organic chemicals.
2. **Local Optimum Plateau:** The initial seed library already contained a high-performing formulation (fluorinated DME, $C_{\text{norm}}^{20} = 0.8168$). Across 7 active-learning batches and 150 candidate rows, the highest newly discovered formulation reached $0.8276$ (+1.3% relative improvement). The primary achievement was structural diversification (identifying cyclic ethers like cyclopentyl methyl ether) rather than finding a dramatically higher peak.

---

# 10. Active-Learning Campaign History

The local dataset preserves the 8-round sequential active learning trajectory:
* **Round 0 (Seed, N=58):** In-house exploratory library; mostly ethers with LiFSI; baseline best target = 0.8168.
* **Round 1 (N=40):** Broad exploration of amines, chlorinated piperidines, non-ethers; severe failure (32/40 dead cells; max = 0.0011).
* **Round 2 (N=23):** Re-anchoring into fluorinated acetals (`COC(OC)C(F)(F)F`); max = 0.4519.
* **Round 3 (N=10):** Polyether chain length tuning; max = 0.5496.
* **Round 4 (N=21):** Hexyl formate and sulfonate esters; max = 0.6343.
* **Round 5 (N=31):** Acetal-glyme hybrids (`COCCOCC(OC)OC`); breakthrough; max = 0.7469.
* **Round 6 (N=16):** Cyclic ethers; discovery of global campaign optimum (`COC1CCCC1` / CPME, $C_{\text{norm}}^{20} = 0.8276$).
* **Round 7 (N=9):** Exploitation/confirmation; 8 of 9 viable; median = 0.7260, max = 0.8200.

---

# 11. Available Experimental Modalities

* **Supported Modalities:** Exactly **ONE** modality: Galvanostatic discharge capacity from coin-cell cycling ($C_{\text{norm}}^{20}$).
* **Absent Modalities:** No electrochemical impedance spectroscopy (EIS), no ionic conductivity ($\sigma$), no cyclic voltammetry stability windows, no viscosity ($\eta$), and no surface characterization (XPS/Raman/NMR).
* **Verdict on Candidate × Measurement Action Space:** **NOT SUPPORTED.** Every experimental action represents the identical full coin-cell assembly and cycling workflow.

---

# 12. Offline Replay Feasibility

We classify offline replay feasibility into a 5-tier taxonomy:

| Replay Category | Status | Detailed Scientific Feasibility |
| :--- | :---: | :--- |
| **1. Historical Trajectory Reconstruction** | **YES** | Fully reconstructs what was known and selected at each historical round (Batches 0..7). |
| **2. Retrospective Next-Batch Ranking** | **YES** | Given data through round $t$, an algorithm can score and rank candidates that were physically tested in batch $t+1$. |
| **3. Finite Historical Label-Pool Replay** | **PARTIAL** | An agent can select among the 208 historically recorded rows and reveal historical outcomes. However, this is a restricted offline lookup, not an open-ended discovery simulation. |
| **4. Counterfactual Closed-Loop Replay** | **NO** | If an agent selects a candidate outside the historical trajectory, no true physical outcome exists. |
| **5. Full 1M Wet-Lab Replay** | **NO** | 99.98% of the 1M candidate library has no experimental measurement. Strict offline replay over the 1M candidate pool is impossible. |

---

# 13. Hypothesis-Space Potential

To prevent over-claiming, scientific explanations are strictly separated into data-supported associations versus literature-informed mechanistic hypotheses:

### Category 1: Data-Supported Empirical Associations
1. **Solvent Functional-Group Correlation:** Ether-containing solvents exhibit statistically higher capacity retention than amines, amides, and esters within the tested library.
2. **Descriptor-Region Clustering:** Top-performing formulations cluster within specific solvent ECFP PCA regions (PCA 0, 1, 4) and solvent MW between 90 and 220 g/mol.
3. **Historical Campaign Concentration:** The active-learning campaign successfully converged towards cyclic and fluorinated ethers over successive rounds.

### Category 2: Literature-Informed Mechanistic Hypotheses (NOT Directly Testable from Dataset Alone)
* *Solvent fluorination promotes LiF-rich inorganic SEI:* While fluorinated ethers perform well, the SEI chemical composition was not measured (no XPS/FTIR).
* *Steric bulk of cyclic ethers weakens Li-solvent binding:* While cyclopentyl methyl ether achieved the highest score, solvation thermodynamics were not measured (no NMR/Raman/MD).
* *LiFSI anion decomposition passivates the anode:* While LiFSI appears in top training rows, the dataset lacks controlled multi-salt comparisons for the best solvents (target values were copied across salts).

---

# 14. Information Leakage & Expansion Audit

* **Candidate Table (`virtual_search_space_1million.csv`):** **SAFE.** Contains no target columns or batch indices.
* **Batch 7 Missingness & Exact Recovery:** Rows 199–207 in `label_all_batches_feat.csv` lacked feature values. All 9 rows were recovered using the exact composite key `(solv_comb_sm, salt_comb_sm)` with **exactly 1 match** in the 1M candidate pool.
* **Target-Copy Expansion (Modeling Table):** **HIGH SIGNIFICANCE.** 115 of 208 rows belong to groups where identical target values were duplicated across different salts. Splitting these rows randomly across train/test folds causes severe data leakage.

---

# 15. Baseline Learnability

We evaluated model learnability across three distinct evaluation protocols:

### Primary Evaluation: Baseline B — Grouped Solvent Cross-Validation (Zero Solvent Overlap)
When grouped by solvent (`GroupKFold(groups=solv_comb_sm)`), train and validation sets have zero chemical solvent overlap, preventing identity leakage:

| Model | MAE | RMSE | $R^2$ Score | Spearman Rank ($\rho$) | Generalization Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Predict Mean)** | 0.2762 | 0.3030 | -0.0175 | -0.1283 | Reference baseline |
| **Ridge Regression ($\alpha=1.0$)** | 0.2337 | 0.2788 | 0.1381 | 0.4660 | Modest linear ranking signal |
| **Random Forest (100 trees)** | 0.2318 | 0.3001 | 0.0015 | 0.3800 | Drops from 0.638 to 0.0015; low unseen-solvent fit |
| **Gaussian Process (Matern5/2)** | **0.1809** | **0.2547** | **0.2809** | **0.5755** | **Robust non-linear ranking signal on unseen chemistry** |

### Reference Evaluation: Baseline A — Row-Wise Cross-Validation (POTENTIAL LEAKAGE)
* Random Forest: MAE = 0.1211, RMSE = 0.1808, $R^2 = 0.6378$, Spearman = 0.7765.
* *Note:* This high score is inflated by pseudo-expanded rows leaking identical solvent targets between folds.

### Temporal Evaluation: Baseline C — Campaign Generalization (Train $\le t$, Test $t+1$)
* Train $\le$ Batch 0 ($N=58$) $\to$ Test Batch 1 ($N=40$): RF Spearman = -0.0684 (Batch 1 exploration failure).
* Train $\le$ Batch 1 ($N=98$) $\to$ Test Batch 2 ($N=23$): RF Spearman = 0.1890.
* Train $\le$ Batch 2 ($N=121$) $\to$ Test Batch 3 ($N=10$): RF Spearman = 0.3450.
* Train $\le$ Batch 3 ($N=131$) $\to$ Test Batch 4 ($N=21$): RF Spearman = 0.3999.
* Train $\le$ Batch 5 ($N=183$) $\to$ Test Batch 6 ($N=16$): RF Spearman = 0.0361.
* Train $\le$ Batch 6 ($N=199$) $\to$ Test Batch 7 ($N=9$): RF Spearman = 0.2333.

---

# 16. Computational Feasibility at 1M Candidates

* Candidate matrix: $999,999 \times 22$ features.
* **Float32 Memory Footprint:** 88.0 MB.
* **Float64 Memory Footprint:** 176.0 MB.
* **Scoring Feasibility:** Chunked scoring in batches of 50,000 rows requires ~8.8 MB per chunk, which is well within standard memory limits.
* *Latency Note:* Actual Bayesian optimization acquisition scoring latency was not benchmarked in this audit and will depend on the surrogate model implementation.

---

# 17. Benchmark Metrics Supported by the Dataset

| Benchmark Metric | Direct Replay ($N=208$) | Surrogate Simulation ($N=1\text{M}$) | Benchmark Role |
| :--- | :---: | :---: | :--- |
| **Max Capacity Found (Best Sample)** | **SUPPORTED** | **SUPPORTED** | Measures discovery peak reached within budget. |
| **Simple Regret ($y^* - \max_{i \le t} y_i$)** | **SUPPORTED** | **SUPPORTED** | Quantifies optimization gap over iteration rounds. |
| **Cumulative Regret** | **SUPPORTED** | **SUPPORTED** | Penalizes selecting dead/failed electrolytes. |
| **Top-K Candidate Recovery** | **SUPPORTED** | **SUPPORTED** | Measures recall of historical high performers. |
| **Predictive Calibration / Log-Likelihood** | **SUPPORTED** | **SUPPORTED** | Evaluates uncertainty quantification quality. |
| **Candidate × Measurement Modality Routing** | **NOT SUPPORTED** | **NOT SUPPORTED** | Single measurement modality only. |

---

# 18. Critical Limitations

1. **Extreme Label Sparsity (0.02% Coverage):** 99.98% of the virtual candidate pool lacks experimental labels.
2. **Single-Modality Action Space:** Absence of intermediate characterizations (conductivity, viscosity, EIS, XPS) prevents multimodal decision-making.
3. **Target-Copy Expansion across Salts:** 115 rows in the ML training table share identical targets across multiple salts, obscuring independent physical experiment counts and preventing unverified salt causal claims.
4. **Severe Chemical Concentration:** >75% of tested formulations are ether derivatives; 414 out of 430 functional classes have zero experimental tests.

---

# 19. Recommended Roles for This Dataset

| Proposed Role | Fit Rating | Detailed Scientific Justification |
| :--- | :---: | :--- |
| **Historical Active-Learning Reproduction Benchmark** | **HIGH FIT** | The 8-round sequential campaign is an exceptional real-world benchmark for testing whether adaptive algorithms can navigate cold-start chemical extrapolation and discover optimal ethers with fewer exploratory failures. |
| **Surrogate Simulation Discovery Benchmark** | **MEDIUM FIT** | Feasible by fitting an in-silico surrogate (Gaussian Process or RF) on the labeled data to simulate the 1M candidate space. However, this is a synthetic benchmark, not historical experimental replay. |
| **Multi-Modal Scientific Reasoning Benchmark** | **LOW FIT** | Unsuitable due to the complete lack of multimodal characterization and staged decision actions. |

---

# 20. Questions Requiring Design Decisions

Before implementing an AIcoScientist domain adapter, the following design decisions must be resolved:

1. **How should candidate selection be constrained in offline benchmarks?**
   * *Option A:* Restrict the offline agent's action space to the 151 pool-compatible historical samples (pure historical replay).
   * *Option B:* Use an in-silico surrogate model as a ground-truth simulator over the full 1M candidate library.
2. **How should the active-learning campaign replay be structured?**
   * Should the benchmark evaluate round-by-round retrospective ranking of batch $t+1$ candidates, or simulate batch selection from the candidate pool?
3. **How should pseudo-expanded salt rows be handled in training?**
   * Should the domain adapter deduplicate target-copied rows to a single unique solvent record, or maintain the author's multi-salt training representation?
4. **How should cathode variations in Batch 0 be normalized?**
   * Should non-LFP cells (12 NMC811 cells and 3 NMC111 cells) be excluded to maintain a strict Cu||LFP benchmark contract?

---

# 21. Required Final Scientific Classification

* **LARGE-SCALE DISCOVERY BENCHMARK:** **MEDIUM FIT** (Feasible only under surrogate oracle simulation, not true 1M wet-lab replay).
* **HISTORICAL ACTIVE-LEARNING RECONSTRUCTION:** **HIGH FIT** (Excellent sequential active-learning benchmark).
* **COUNTERFACTUAL CLOSED-LOOP WET-LAB REPLAY:** **NOT SUPPORTED** (Cannot reveal true unmeasured physical outcomes).
* **MULTIMODAL SCIENTIFIC REASONING:** **LOW FIT** (Lacks multimodal experimental measurements).
* **CANDIDATE × MEASUREMENT DECISION SPACE:** **NOT SUPPORTED** (Single experimental modality only).

---
*End of Corrected Dataset Audit Report.*

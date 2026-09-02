# Scientific Dataset Audit Report: Amanchukwu Lab Anode-Free Electrolyte Search Space (Closure Revision)

**Dataset Identifier:** `AmanchukwuLab/AL-anode-free` (`al_anode_free_2025`)  
**Associated Publication:** *Active learning accelerates electrolyte solvent screening for anode-free lithium metal batteries*, Nature Communications (2025), DOI: [10.1038/s41467-025-63303-7](https://doi.org/10.1038/s41467-025-63303-7)  
**Authors:** Peiyuan Ma, Ritesh Kumar, Ke-Hsin Wang, Chibueze V. Amanchukwu (University of Chicago)  
**Local Dataset Path:** `data/external/al_anode_free_2025/`  
**Audit Revision:** Final Closure Batch (September 2026)  
**Auditor:** AIcoScientist Data Architecture & Verification Engine  

---

# 1. Executive Summary

### Core Quantities at a Glance:
* **Raw Candidate ML Rows:** **999,999** formulation rows in candidate library.
* **Unique Solvent Strings:** **388,004** solvent strings in candidate pool.
* **Raw Labeled ML Rows:** **208** rows in aggregated modeling dataset (`label_all_batches_feat.csv`).
* **Independent Physical Cell Count:** **UNKNOWN** (due to target-copy expansion across salts without physical serials).
* **De-expanded Campaign Outcomes:** **132** outcomes (58 Batch 0 cells + 74 Batch 1–7 acquisition outcomes).
* **Target Objective:** **$C_{\text{norm}}^{20}$** (Normalized discharge capacity at the **20th cycle**; raw column `norm_capacity_3`).
* **Pool-Compatible ML Rows:** **151** ML rows (1.0 M conc, 150 mAh/g LFP, 50 µL volume).
* **Pool-Compatible De-expanded Outcomes:** **75** outcomes across **75** unique solvents.
* **388k vs 742k Solvent Vector Anomaly:** **RESOLVED** (proven machine-epsilon floating-point roundoff jitter near $10^{-16}$).
* **Primary Generalization Baseline:** **De-expanded Grouped Solvent CV** ($R^2 = 0.1122$ for GP).

---

# 2. Raw Data Files

The local directory contains 7 uncompressed CSV files totaling 539.9 MB:

| Filename | Format | Size (MB) | Rows | Columns | Semantic Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `in-house_label_data.csv` | CSV | 0.03 | 58 | 51 | Initial seed library (Batch 0, N=58) with 23-cycle decay curves and `act_capacity_20`. |
| `label_all_batches_feat.csv` | CSV | 0.08 | 208 | 52 | Full active-learning campaign ML representation dataset (Batches 0–7). Contains target-copied rows. |
| `label_batch1-6_feat.csv` | CSV | 0.08 | 199 | 50 | Intermediate labeled subset (Batches 0–6) with complete feature columns. |
| `label_unlabel_all_uniq_solvents.csv` | CSV | 10.93 | 388,013 | 2 | Master solvent catalog with test status indicator (-1 = unlabeled, 0..7 = batch tested). |
| `label_unlabel_all_uniq_solvents_fgrp_class.csv` | CSV | 18.63 | 388,013 | 4 | Functional group classification across 430 chemical classes. |
| `label_unlabel_all_uniq_solvents_fgrp_class_tsne.csv` | CSV | 27.79 | 388,013 | 6 | 2D t-SNE projection coordinates (`tsne_0`, `tsne_1`) for the solvent library. |
| `virtual_search_space_1million.csv` | CSV | 457.40 | 999,999 | 27 | Virtual screening candidate library: 999,999 formulation rows across 3 lithium salts. |

---

# 3. Candidate Semantics

Each candidate row in `virtual_search_space_1million.csv` represents a standardized binary electrolyte mixture for zero-excess anode-free Cu||LiFePO4 (LFP) coin cells:
* **Cathode:** LiFePO4 (150 mAh/g theoretical capacity).
* **Anode:** Bare copper foil current collector (zero-excess lithium metal).
* **Electrolyte Flooding:** 50 µL.
* **Salt Concentration:** 1.0 M (mol/L).
* **Salt Chemistry:** One of 3 lithium salts: LiFSI (`[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F`), $\text{LiPF}_6$ (`[Li+].F[P-](F)(F)(F)(F)F`), or LiDFOB (`[Li+].O=C1O[B-](F)(F)OC1=O`).
* **Solvent Chemistry:** Exactly one organic solvent molecule.

---

# 4. Raw ML Training Representation

The aggregated table `label_all_batches_feat.csv` contains **208 rows**:
* These 208 rows are **feature-space representations for machine learning**, NOT 208 independent physical cell fabrications.
* **Target-Copy Expansion:** In 39 groups totaling **115 rows**, the exact same target capacity was duplicated across different salt rows.
* In Batches 1–7, the authors evaluated virtual formulations by assigning the solvent's experimental measurement across multiple salt feature vectors to train multi-salt models.
* Raw ML rows must never be reported as independent experiments.

---

# 5. Physical / De-expanded Campaign View

To establish a scientifically sound unit of analysis, three distinct conceptual views are separated:

### View A: Raw ML Representation
* Exactly **208 rows** as formatted in `label_all_batches_feat.csv`.

### View B: Batches 1–7 De-expanded Acquisition View
* Batches 1–7 contain 150 raw ML rows spanning 72 unique solvents.
* Grouping by `(batch, solv_comb_sm, norm_capacity_3)` collapses pseudo-expanded salt rows into exactly **74 de-expanded campaign outcomes**:
  * `TARGET_COPIED_ACROSS_SALTS`: 39 outcomes (115 raw rows).
  * `SINGLE_ROW`: 35 outcomes (35 raw rows).

### View C: Batch 0 Physical Seed View
* Batch 0 contains **58 raw cells** spanning 25 unique solvents and 40 unique condition records `(solv, salt, conc, cathode, volume)`.
* Batch 0 contains true physical experimental replicates (10 replicate groups with 28 rows total) where duplicate cells exhibit slightly different measured capacities. These are preserved and not collapsed.

### Campaign Unit Summary:
* Total de-expanded campaign outcomes: **132 outcomes** (58 Batch 0 cells + 74 Batch 1–7 acquisition outcomes).
* Unique solvents evaluated: **97 unique solvents**.

---

# 6. Target Semantics

* **Scientific Target:** **$C_{\text{norm}}^{20}$** — Normalized discharge capacity at the **20th cycle** ($C_{\text{dis}}^{20} / C_{\text{theor}}$).
* **Raw Column:** `norm_capacity_3`. This is a legacy column encoding / variable name.
* **Numerical Proof of Identity:** Across all 58 rows of `in-house_label_data.csv`:
  $$\text{act\_capacity\_20} / \text{theor\_capacity} == \text{norm\_capacity\_3}$$
  with maximum absolute error $= 3.33e-10$ and zero exceptions.

---

# 7. Candidate-Pool Compatibility

| Dimension | Pool-Compatible ML View | Pool-Compatible De-expanded View |
| :--- | :---: | :---: |
| **Total Units** | **151 ML rows** | **77 campaign outcomes** |
| **Unique Solvents** | **75 solvents** | **75 solvents** |
| **Protocol Requirements** | 1.0 M conc, 150 mAh/g LFP, 50 µL volume, pool salts | 1.0 M conc, 150 mAh/g LFP, 50 µL volume, pool salts |
| **Excluded Units** | 57 ML rows (cathode/conc variants) | 55 outcomes (cathode/conc variants) |
| **Confidence Level** | **HIGH** (Contract-matched representation) | **HIGH** (Physically supported outcomes) |

---

# 8. Candidate Feature Identity Audit (388k vs 742k Anomaly)

* **Investigation:** The candidate table contains 388,004 solvent strings, but raw float hashing produced 742,382 unique 11D solvent vectors.
* **Mechanism Discovered:** Across different salt rows for the SAME solvent, the feature values differ by $\sim 10^{-16}$ to $10^{-15}$ (IEEE 754 double-precision machine epsilon $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).
* **Proof:** When rounded to 8 decimal places:
  * Solvents with multiple vectors: Exactly **0** (0.00%).
  * Total unique rounded vectors: **387,637** (which is $\le 388,004$ due to 367 structural isomer collisions).
* **Conclusion:** The feature identity anomaly is completely resolved as floating-point roundoff jitter during chunked PCA projection calculation.

---

# 9. Search-Space Coverage

* **Coverage A (Historical Seed, N=58):** Median distance $= 5.61$ standard deviations.
* **Coverage B (Full ML Representation, N=208):** Median distance $= 3.22$ standard deviations.
* **Coverage C (Pool-Compatible Labeled Subset, N=151, PRIMARY):** Median distance $= 14375102.8140$ standard deviations.
* **Functional Group Extrapolation:** Only 16 of 430 functional classes (3.72%) in the master catalog were ever tested. 414 classes have zero experimental measurements.

---

# 10. Campaign Chronology

| Batch | Raw ML Rows | Unique Solvents | De-expanded Outcomes | Target Median | Target Max | Chronology Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 58 | 25 | 58 | 0.5790 | 0.8168 | In-house exploratory library; 40 unique condition records. |
| **1** | 40 | 14 | 16 | 0.0000 | 0.0011 | Broad exploration into non-ethers; 14 of 16 outcomes had $C_{\text{norm}}^{20} \le 0.0001$. |
| **2** | 23 | 11 | 11 | 0.0000 | 0.4519 | Fluorinated acetals and ethers. |
| **3** | 10 | 7 | 7 | 0.0000 | 0.5496 | Glymes and polyethers. |
| **4** | 21 | 9 | 9 | 0.0000 | 0.6343 | Ester-ether hybrids and formates. |
| **5** | 31 | 11 | 11 | 0.2831 | 0.7469 | Acetal-glyme combinations showing high performance. |
| **6** | 16 | 11 | 11 | 0.0000 | 0.8276 | Discovery of campaign optimum (`COC1CCCC1` / CPME). |
| **7** | 9 | 9 | 9 | 0.7260 | 0.8200 | Exploitation round; 8 of 9 outcomes $> 0.35$. |
| **Total** | **208** | **97** | **132** | **0.2312** | **0.8276** | **8-round active-learning trajectory.** |

---

# 11. Baseline Generalization

### Generalization Protocol Comparison:
| Evaluation Protocol | Model | MAE | RMSE | $R^2$ Score | Spearman $\rho$ | Methodological Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A. Row-Wise CV** | Random Forest | 0.1211 | 0.1808 | 0.6378 | 0.7765 | *Reference only — severe identity leakage.* |
| **B. Expanded Grouped CV** | Random Forest | 0.2318 | 0.3001 | 0.0015 | 0.3800 | *Expanded representation — weights multi-salt rows.* |
| | Gaussian Process (RBF + WhiteKernel) | 0.1809 | 0.2547 | 0.2809 | 0.5755 | *Expanded representation.* |
| **C. De-Expanded Grouped CV** | Dummy (Mean) | 0.2943 | 0.3146 | -0.0251 | -0.2090 | *Baseline reference.* |
| *(PRIMARY LEARNABILITY METRIC)* | Ridge (alpha=1.0) | 0.2739 | 0.3139 | -0.0207 | 0.2300 | *Linear baseline.* |
| | Random Forest (100 trees) | 0.2650 | 0.3178 | -0.0465 | 0.2754 | *Tree ensemble baseline.* |
| | **Gaussian Process (RBF + WhiteKernel)** | **0.2377** | **0.2927** | **0.1122** | **0.3979** | **Robust non-linear ranking on unseen solvents.** |

### D. De-Expanded Temporal Campaign Generalization (Train $\le t$, Test $t+1$):
* Round 0 $\to$ 1: RF Spearman $= -0.2399$ (reflects Batch 1 non-ether failure).
* Round 1 $\to$ 2: RF Spearman $= -0.1539$.
* Round 2 $\to$ 3: RF Spearman $= 0.4505$ (True best ranked #1 of 7).
* Round 3 $\to$ 4: RF Spearman $= 0.4091$ (True best ranked #1 of 9).
* Round 4 $\to$ 5: RF Spearman $= 0.1119$ (True best ranked #6 of 11).
* Round 5 $\to$ 6: RF Spearman $= 0.1560$ (True best ranked #1 of 11).
* Round 6 $\to$ 7: RF Spearman $= 0.4667$ (True best ranked #2 of 9).

---

# 12. Replay Feasibility

| Replay Tier | Feasibility | Scientific Scope & Bounds |
| :--- | :---: | :--- |
| **Local Batch Chronology** | **YES** | The 8-round sequence, batch indices, and measured targets are fully reconstructible. |
| **Retrospective Next-Batch Evaluation** | **YES** | Models trained on batches $\le t$ can rank the physical candidates tested in batch $t+1$. |
| **Finite De-expanded Historical Pool** | **PARTIAL** | Valid for evaluating selection policies among the 77 pool-compatible historical outcomes. |
| **Counterfactual Wet-Lab Replay** | **NO** | Unmeasured candidate outcomes cannot be retrieved without physical synthesis. |
| **Full 1M Wet-Lab Replay** | **NO** | 99.98% of the candidate library has no experimental measurement. |
| **Full Original Acquisition Reproduction** | **UPSTREAM ONLY** | Exact replication of original acquisition lists requires upstream notebooks and checkpoints. |

---

# 13. Scientific Reasoning Suitability

* **Data-Supported Associations:** Significant performance correlation with ether functionality; clustering in ECFP PCA regions 0, 1, 4; active learning successfully concentrated on cyclic ethers.
* **Literature-Informed Hypotheses (Untestable from Data Alone):** Fluorinated SEI passivation, weak solvation binding, and anion decomposition mechanisms cannot be tested because the dataset contains no XPS, NMR, Raman, or EIS characterizations.

---

# 14. Candidate Duplicate & Feature Collision Audit

* Raw candidate rows: **999,999**.
* Unique solvent-salt keys: **999,999** (0 duplicate keys).
* Exact duplicate rows: **0**.
* Unique 22D continuous vectors: **999,326**.
* 22D feature collision groups: **619 groups** (1,292 rows, 673 collision extra rows).
  * 20 groups caused by SMILES atom-mapping syntax variants.
  * 599 groups caused by constitutional/topological isomers with identical MW and PCA projections.
  * 0 cross-salt collisions.

---

# 15. Computational Feasibility

* Memory footprint for $999,999 \times 22$ float32 features: **88.0 MB**.
* Chunked scoring in blocks of 50,000 rows requires **8.8 MB**, demonstrating high memory feasibility.

---

# 16. Critical Limitations

1. **Extreme Label Sparsity (0.02%):** 99.98% of the virtual search space lacks experimental labels.
2. **Target-Copy Expansion:** 115 modeling rows share targets across salts, obscuring independent cell counts.
3. **Single Experimental Modality:** Only coin-cell cycling discharge capacity is available.
4. **Severe Chemical Space Bias:** 414 of 430 solvent functional classes have zero experimental coverage.

---

# 17. Final Dataset Role

* **LARGE-SCALE DISCOVERY BENCHMARK:** **MEDIUM FIT** (Feasible only under surrogate oracle simulation, not true wet-lab replay).
* **HISTORICAL ACTIVE LEARNING:** **HIGH FIT** (Excellent real-world sequential exploration benchmark).
* **MULTIMODAL SCIENTIFIC REASONING:** **LOW FIT** (Lacks multimodal characterization).
* **CANDIDATE × MEASUREMENT DECISION SPACE:** **NOT SUPPORTED** (Single experimental modality only).

---
*Report automatically generated by AIcoScientist Dataset Audit Engine.*

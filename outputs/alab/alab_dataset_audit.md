# A-Lab Precursor Genome Dataset Intake & Scientific Affordance Audit

**Audit Date**: 2026-09-01T15:08:29.609364+00:00  
**Dataset Name**: A-Lab Precursor Genome  
**Local Root**: `data/external/precursor_genome_2026`  
**License**: CC BY 4.0  
**Source / Zenodo**: [https://doi.org/10.5281/zenodo.21285546](https://doi.org/10.5281/zenodo.21285546)  

---

## 1. Candidate Identity & Search Space

- **Primary Identity Key**: `sample_id` (`PG_0001` through `PG_4450`)
- **Total Experimental Candidates**: `1035` (100% unique primary keys, zero unindexed duplicates)
- **Precursor Formulation**: Exactly 2 binary precursors per reaction chosen from a library of **46** unique stoichiometric precursor compounds.
- **Unique Target Compounds**: **1032** distinct inorganic target compositions.

---

## 2. Information Firewall Classification

To enforce strict, leakage-free offline replay without lookahead bias:

| Classification | Fields / Modalities | Description |
|---|---|---|
| **PRE_EXPERIMENT** (Visible before execution) | `sample_id`, `precursor_1`, `precursor_2`, `target_compound`, `heating_temperature_c`, `heating_time_minutes`, `reaction_energy_ev_per_atom` | Candidate identity, thermodynamic reaction energy, nominal formulation, and controllable furnace heating parameters. |
| **HIDDEN / REVEALABLE** (Available only via scientific actions) | `raw_scans.zip` (`XRD`), `refinement_pkls.zip` (`REFINEMENT`), `outcome.reaction_category` (`OUTCOME_TEST`) | Post-reaction crystal structure diffraction patterns, Rietveld phase quantification, and synthesis conversion outcomes. |
| **PRECURSOR CHARACTERIZATION** | `sem.zip` (`SEM_PRECURSOR`), `eds.zip` (`EDS_PRECURSOR`) | Pre-reaction precursor morphology and elemental quantification tables. |

---

## 3. Experimental Modality Inventory

| Modality | Archive / Source | Coverage | Normalized Cost | Representation & Prerequisite |
|---|---|---|---|---|
| **`XRD`** | `raw_scans.zip` (1,419 members) | **100.0%** (1035/1,035) | `1.0` | High-resolution 2theta intensity vector standardized onto 450-point grid; PCA basis fitted strictly on revealed historical spectra and frozen during Bayesian updates. |
| **`REFINEMENT`** | `refinement_pkls.zip` (2,964 members) | **99.52%** (1030/1,035) | `0.5` | Extracted Rietveld phase weights, Rwp goodness-of-fit, and target formation fraction. **Requires `XRD` action first**. |
| **`OUTCOME_TEST`** | `ledger_precursor_genome.json` | **100.0%** (1,035/1,035) | `2.0` | Quantitative reaction conversion score in $[0.0, 1.0]$. |

---

## 4. Primary Discovery Objective

- **Objective**: `reaction_conversion`
- **Direction**: `MAXIMIZE`
- **Scientific Meaning**: Quantitative extent of solid-state reaction conversion, mapping synthesis outcomes to continuous performance metrics:
  - `completely_reacted`: $1.0$ (143 samples, 13.8%)
  - `transformed`: $0.75$ (384 samples, 37.1%)
  - `partially_reacted`: $0.5$ (113 samples, 10.9%)
  - `unreacted`: $0.0$ (369 samples, 35.7%)

---

## 5. Local Archive Integrity & Statistics

| Archive Name | Member Count | Compressed Size | Uncompressed Size | Linkage Status |
|---|---|---|---|---|
| `raw_scans.zip` | 1419 | 21.94 MB | 56.83 MB | 100% linked to sample IDs |
| `refinement_pkls.zip` | 2964 | 277.91 MB | 862.29 MB | 99.5% linked to sample IDs |
| `sem.zip` | 342 | 389.28 MB | 393.68 MB | Precursor library reference |
| `eds.zip` | 242 | 350.17 KB | 1.44 MB | Precursor library reference |

---

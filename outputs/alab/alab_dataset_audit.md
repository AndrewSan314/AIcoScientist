# A-Lab Precursor Genome Dataset Scientific Audit

**Audit Date**: 2026-09-01T18:33:21.527569+00:00  
**Dataset**: A-Lab Precursor Genome (`precursor_genome_2026`)  
**License**: CC BY 4.0  
**Primary Key**: `sample_id`  

---

## 1. Candidate Population & Chemical Coverage

- **Total Unique Candidates**: 1035 (100% unique sample IDs)
- **Unique Chemical Precursors**: 0 (Canonical multi-hot encoding basis: 46 binary flags)
- **Unique Target Compounds**: 1032

---

## 2. Information Firewall & Modality Structure

| Modality Name | Physical Meaning | Cost | Availability | Coverage | Requirements |
|---|---|---|---|---|---|
| `XRD` | Powder X-ray diffraction (450-pt 10–100° $2\theta$ grid) | 1.0 | 1035 / 1035 | 100.0% | None |
| `REFINEMENT` | Rietveld phase weights & Rwp | 0.5 | 1030 / 1035 | 99.52% | `XRD` |
| `OUTCOME_TEST` | Synthesis outcome utility (0.0–1.0) | 2.0 | 0 / 1035 | 0.0% | None |

---

## 3. Outcome Distribution & Missingness Handling

| Outcome Category | Utility Score | Count | Percentage |
|---|---|---|---|
| `completely_reacted` | 1.00 | 0 | 0.0% |
| `transformed` | 0.75 | 0 | 0.0% |
| `partially_reacted` | 0.50 | 0 | 0.0% |
| `unreacted` | 0.00 | 0 | 0.0% |
| `unlabeled` (physical failures) | `None` (no silent 0 imputation) | 1035 | 100.0% |

> [!NOTE]
> Unlabeled physical failures (e.g., tube breaks, robotic dispensing aborts) reveal `reaction_outcome_utility = None` and `canonical_observation = None`. They are not silently imputed as 0.0 unreacted.

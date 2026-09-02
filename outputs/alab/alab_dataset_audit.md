# A-Lab Precursor Genome Real Dataset Audit

**Audit Timestamp**: 2026-09-02T04:10:46.024879+00:00  
**Source Dataset**: A-Lab Precursor Genome (`precursor_genome_2026`)  
**Local Path**: `data/external/precursor_genome_2026`  

## 1. Candidate Population & Chemical Identities

- **Total Candidates**: 1035 (100% unique primary keys)
- **Unique Precursors in Dataset**: 46
- **Canonical Precursor Feature Dimension**: 46 (one-hot vector)
- **Unique Target Compounds**: 1032

## 2. Reaction Outcome Semantics & Labeled Coverage

- **Classified Synthesis Outcomes**: 1009 (97.5%)
- **Unclassified Outcomes (Physical Failures / Missing)**: 26 (2.5%)

| Reaction Category | Count | Percentage | Utility Value |
|---|---|---|---|
| `completely_reacted` | 143 | 13.8% | 1.00 |
| `transformed` | 384 | 37.1% | 0.75 |
| `partially_reacted` | 113 | 10.9% | 0.50 |
| `unreacted` | 369 | 35.7% | 0.00 |
| `unclassified` | 26 | 2.5% | None (Filtered / Fail-Closed) |

## 3. Physical Characterization Data Coverage

- **Total Raw XRD Scans**: 1351 across 1035 samples (0 samples with 0 scans)
- **Active Scan Index Distribution**: {0: 900, 1: 108, 2: 22}
- **Total Rietveld Refinement Cases**: 1950 across 1030 samples
- **Phase Weight Unit Normalization**: 0 percentage-scale cases normalized to fractional scale

## 4. Information Firewall Compliance

Pre-experiment candidate representation consists strictly of reaction thermodynamics, heating conditions, and one-hot precursor presence.
Post-synthesis measurements (XRD scans, Rietveld structural refinements, and reaction outcome utilities) are strictly isolated behind the experimental oracle.

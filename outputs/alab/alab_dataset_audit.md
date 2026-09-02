# A-Lab Precursor Genome Real Dataset Audit

**Audit Timestamp**: 2026-09-02T06:43:53.068360+00:00  
**Source Dataset**: A-Lab Precursor Genome (`precursor_genome_2026`)  
**Local Path**: `data/external/precursor_genome_2026`  

## 1. Candidate Population & Chemical Identities

- **Total Candidates**: 1035 (100% unique primary keys)
- **Unique Precursors in Dataset**: 46
- **Canonical Precursor Feature Dimension**: 46 (one-hot vector)
- **Unique Target Compounds**: 1032

## 2. Reaction Outcome Semantics & Labeled Coverage

- **Classified Synthesis Outcomes**: 1009 (97.5%)
- **Unclassified Outcomes (Missing Reaction Categories)**: 26 (2.5%)
- **Physical Failure Flag Presence**: 26 samples confirmed with `phases_unavailable_reason: 'physical_failure'` in raw ledger

| Reaction Category | Count | Percentage | Utility Value |
|---|---|---|---|
| `completely_reacted` | 143 | 13.8% | 1.00 |
| `transformed` | 384 | 37.1% | 0.75 |
| `partially_reacted` | 113 | 10.9% | 0.50 |
| `unreacted` | 369 | 35.7% | 0.00 |
| `unclassified` | 26 | 2.5% | None (Filtered / Fail-Closed) |

## 3. Physical Characterization Data Coverage & Canonical Usability

- **Total Raw XRD Scans**: 1351 across 1035 samples (0 samples with 0 scans)
- **Canonical XRD Resolvable & Usable for Replay**: 1035 / 1035 (100.0%)
- **Canonical XRD Selection Methods**: {'ledger_active_scan_index': 1030, 'status_active_or_valid': 5}
- **Canonical Rietveld Refinements Usable for Replay**: 1030 / 1035 (99.5%)
- **Refinement Source Breakdown**: 1030 structured ledger phase weights, 0 pickle artifacts, 5 missing
- **Canonical Refinement Selection Methods**: {'ledger_active_case_index': 1030, 'no_refinement_cases': 5}
- **Refinement Origin**: 364 manual, 666 automated
- **Phase Weight Unit Scale**: Phase weights were validated for unit scale; all observed A-Lab ledger refinement weights in this dataset version were fraction-scale. The parser also supports percentage-scale normalization defensively.

## 4. Information Firewall Compliance

Pre-experiment candidate representation consists strictly of reaction thermodynamics, heating conditions, and one-hot precursor presence.
Post-synthesis measurements (XRD scans, Rietveld structural refinements, and reaction outcome utilities) are strictly isolated behind the experimental oracle.

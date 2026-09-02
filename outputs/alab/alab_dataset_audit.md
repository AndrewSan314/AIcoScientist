# A-Lab Precursor Genome Real Dataset Audit

**Audit Timestamp**: 2026-09-02T16:29:57.783345+00:00  
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
- **Physical Failure Records**: 26 samples with `physical_failure` metadata
- **Phase Data Unavailable Due to Physical Failure**: 26 samples confirmed with `phases_unavailable_reason: 'physical_failure'`
- **Unclassified and Physical Failure**: 26
- **Unclassified without Physical Failure**: 0

| Reaction Category | Count | Percentage | Utility Value |
|---|---|---|---|
| `completely_reacted` | 143 | 13.8% | 1.00 |
| `transformed` | 384 | 37.1% | 0.75 |
| `partially_reacted` | 113 | 10.9% | 0.50 |
| `unreacted` | 369 | 35.7% | 0.00 |
| `unclassified` | 26 | 2.5% | None (Filtered / Fail-Closed) |

## 3. Physical Characterization Data Coverage & Canonical Usability

- **Total Raw XRD Scans**: 1351 across 1035 samples (0 samples with 0 scans)
- **Canonical Scans vs Replay Fallbacks**:
  - Ledger-canonical active scans: 1030
  - Upstream-recomputed canonical scans: 0
  - Deterministic replay-only fallback scans: 5
  - Total replayable XRD: 1035 / 1035 (100.0%)
  - Unusable XRD: 0
- **Canonical XRD Selection Methods**: {'ledger_active_scan_index': 1030, 'replay_fallback_valid_scan': 5}
- **XRD XML Parsable**: 1035 (0 malformed)
- **Physical 2Theta Axis Extraction**: 1035 from XML positions, 0 from ledger settings, 0 missing
- **XRD Intensity Counts**: 1035 valid, 0 missing
- **Canonical Rietveld Refinements Usable for Replay**: 1030 / 1035 (99.5%)
- **Refinement Source Breakdown**: 1030 structured ledger phase weights, 0 pickle artifacts, 5 missing
- **Canonical Refinement Selection Methods**: {'ledger_active_case_index': 1030, 'no_refinement_cases': 5}
- **Refinement Origin**: 364 manual, 666 automated
- **Phase Weight Unit Scale**: Phase weights were validated for unit scale; all observed A-Lab ledger refinement weights in this dataset version were fraction-scale. The parser also supports percentage-scale normalization defensively.

## 4. Information Firewall Compliance

Pre-experiment candidate representation consists strictly of reaction thermodynamics, heating conditions, and one-hot precursor presence.
Post-synthesis measurements (XRD scans, Rietveld structural refinements, and reaction outcome utilities) are strictly isolated behind the experimental oracle.

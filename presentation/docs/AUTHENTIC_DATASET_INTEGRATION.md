# AIcoScientist — Authentic Scientific Dataset Registry & Resolver Architecture

**Document Version:** 1.0.0  
**Status:** Implemented & Verified  
**Target Repository:** `AndrewSan314/AIcoScientist`  
**Integration Branch:** `integration/multimodal-scientific-engine`  

---

## 1. Executive Summary

Historically, presentation demonstrations frequently relied on ad-hoc synthetic presets, hardcoded multipliers, or fabricated sample counts. In AIcoScientist, **all fabricated, synthetic fallback, or mislabeled datasets have been eliminated** from the presentation layer.

In their place, we have established:
1. **A Canonical Scientific Dataset Registry** (`presentation/data/dataset_registry.json`), embedded directly into `snapshot.json` (schema `1.2.0`) and verified against raw source artifacts via SHA-256 manifests.
2. **A Pure Campaign Resolver** (`presentation/frontend/src/utils/campaignResolver.ts`), which dynamically maps dataset selections into fully typed, mathematically consistent campaign views, isolating candidate identity pools and ensuring strict adherence to physical capabilities.
3. **Automated Integrity & Isolation Tests** (`presentation/tests/test_ui_integrity.py`), asserting zero fabricated numbers, exact sample counts, and complete isolation between dataset candidate spaces.

---

## 2. The Three Authentic Scientific Benchmarks

| Metric / Dimension | 1. Controlled Multimodal Alloy | 2. A-Lab Precursor Genome | 3. Anode-Free Electrolyte Screening |
| :--- | :--- | :--- | :--- |
| **Canonical Dataset ID** | `controlled_multimodal_alloy` | `alab_precursor_genome` | `anode_free_electrolyte_screening` |
| **Legacy UI Alias** | `controlled_synthesis` | `alab_replay` | `electrolyte_search` |
| **Benchmark Type** | `IN_SILICO_BENCHMARK` | `HISTORICAL_LAB_REPLAY` | `SURROGATE_SCREENING` |
| **Scientific Domain** | In-Silico Controlled Solid-State Worlds | Autonomous Solid-State Inorganic Synthesis | High-Entropy LiFSI Liquid Battery Electrolytes |
| **Primary Source Artifacts** | `outputs/alab/multimodal/clean_controlled_worlds.json`<br>`outputs/alab/multimodal/full_policy_matrix.json` | `data/external/precursor_genome_2026/ledger_precursor_genome.json`<br>`outputs/alab/multimodal/evidence_ledger.jsonl` | `data/external/al_anode_free_2025/`<br>`outputs/electrolyte/benchmark/screening_quality_diagnostics.json`<br>`outputs/electrolyte/benchmark/surrogate_simulation.json` |
| **Candidate Count** | **12** candidates (`controlled-0` to `controlled-11`) | **1,035** physical samples (`PG_0102` to `PG_1521`) | **333,333** virtual formulations (WS = 200) |
| **Characterization Modalities** | • XRD (diagnostic, cost 1.0)<br>• Rietveld (diagnostic, cost 2.0)<br>• Outcome (evaluative, cost 2.0) | • XRD (active, cost 1.0)<br>• Rietveld (active, cost 2.0)<br>• SEM (unlinked, unavailable)<br>• EDS (unlinked, unavailable) | • Screening Filter (rank ensemble, cost 0.0)<br>• Surrogate Oracle (ExtraTrees, cost 1.0) |
| **Target Observable** | Multi-hypothesis state vectors: Phase Purity ($H_1$), Composition ($H_2$), Morphology Kinetics ($H_3$) | Synthesis synthesizability & phase classification | Cycle-3 normalized discharge capacity (`norm_capacity_3`, NOT conductivity) |
| **Default Trajectory** | `policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID`<br>(4 steps, 6.0 credits total cost) | `replay:HYBRID:42:1`<br>(6 steps, PG_0309, PG_0214, PG_0209) | `HYBRID_DEFAULT`<br>(15 sequential surrogate evaluations) |
| **Provenance & Citation** | AIcoScientist In-Silico Benchmark<br>(180 runs: 6 worlds × 5 seeds × 6 policies) | Walters et al., A-Lab Precursor Genome<br>(Zenodo DOI: `10.5281/zenodo.21285546`, CC BY 4.0) | AmanchukwuLab, *Nature Communications* 2025<br>(DOI: `10.1038/s41467-025-63303-7`, CC BY 4.0) |
| **Key Disclosures** | Synthetic benchmark for formal Bayesian bounds and counterfactual comparison | Replay is locked to historical sequence; SEM/EDS unlinked; Gate 17 over-dispersed | Surrogate oracle is univariate ExtraTrees in-silico approximation, not wet-lab cycling |

---

## 3. Dataset Registry Schema Specification

The dataset registry is defined in `presentation/frontend/src/types/mission_control.ts` and encoded in `presentation/data/dataset_registry.json`. Each entry conforms to:

```typescript
export interface ScientificDatasetRegistryEntry {
  id: CanonicalDatasetId;
  legacy_id?: LegacyDatasetId;
  displayName: string;
  domain: string;
  provenance: DatasetProvenance;
  candidateCount: number;
  screenedWorkingSetCount?: number;
  candidateIds?: string[];
  targetObservable?: string;
  targetObservableDescription?: string;
  modalities: Record<string, DatasetModalityInfo>;
  hypotheses: string[];
  defaultConfiguration: Record<string, any>;
  capabilities: DatasetCapabilities;
  summary: string;
  statusBadge: string;
  disclosures: string[];
}
```

### Critical Capability Flags
- `competingHypotheses`: `true` only for `controlled_multimodal_alloy` where Bayesian inference adjudicates among mutually exclusive hypotheses ($H_1, H_2, H_3$). Set to `false` for A-Lab and Electrolyte.
- `candidateScreening`: `true` only for `anode_free_electrolyte_screening` where a 4-tranche rank ensemble screens 333,333 virtual formulations down to a working set of 200 in 2.535s.
- `preregistrationReplay`: `true` for Controlled Alloy and A-Lab Replay, where audit records enforce pre-reveal hypothesis distributions and state sequence ordering ($t_B < t_C < t_D$).
- `closedLoopExecution`: `true` when the policy actively selects subsequent actions from the unmeasured pool; `false` in A-Lab because the sequence was historically determined in the robotic wet lab.
- `surrogateSimulation`: `true` only for Electrolyte screening, where the measurement engine is a trained ExtraTrees regressor.

---

## 4. Campaign Resolver Architecture (`campaignResolver.ts`)

The campaign resolver provides a clean, fail-closed functional layer that separates UI view components from the raw snapshot structure:

```
[UI Workspace (Discovery Lab)]
       │
       ▼
resolveCampaign(selectedDatasetId, snapshotData)
       │
       ├── Case 1: "controlled_multimodal_alloy"
       │     └── Maps flagship_campaign (4 steps, 12 candidates, 3 hypotheses, full 24-action counterfactuals)
       │
       ├── Case 2: "alab_precursor_genome"
       │     └── Maps replay_campaign (6 steps, 1035 samples, authentic Rietveld/XRD observations)
       │
       ├── Case 3: "anode_free_electrolyte_screening"
       │     └── Synthesizes sequential surrogate query steps from electrolyte_simulation (15 queries)
       │
       └── Returns ResolvedCampaignView
```

### Observation Blinding Firewall
For every resolved step:
- **State A (Observation Selection)**: Policy ranks candidate actions using prior beliefs.
- **State B (Preregistered Decision)**: Policy commits to the optimal action $a^*$. Predictive distributions are logged. Observed measurements remain strictly hidden.
- **State C (Revealed Measurement)**: Actual sensor observation (XRD intensity, Refinement $R_{wp}$, or cycle-3 capacity) is revealed.
- **State D (Posterior Update)**: Bayesian or surrogate posterior update is applied, reducing entropy.

---

## 5. UI Integration & View Specialization

In `presentation/frontend/src/views/DiscoveryLabWorkspace.tsx`:
1. **Interactive Setup Controls**:
   - Selecting **A-Lab Precursor Genome** automatically locks policy to `HYBRID` and explains: *"Fixed Historical Laboratory Replay — Policy selection is disabled because experimental sequence was historically fixed in physical laboratory."*
   - Selecting **Electrolyte Screening** displays the 333,333 virtual pool, the 200 screened working set, and targets `norm_capacity_3`.
2. **Dynamic Top Banner**: Displays authentic metadata, candidate counts, provenance badges, and disclosures matching the active dataset.
3. **Step Buttons & Progression**:
   - Steps 1–4 for Controlled Alloy.
   - Steps 1–6 for A-Lab Replay (PG_0309, PG_0214, PG_0209).
   - Iterations 1–15 for Electrolyte Surrogate Screening.
4. **Lead Visualizations**:
   - Competing hypotheses trajectories and 3D Stark Hologram for Controlled Alloy.
   - Sample-level XRD/Refinement evidence cards for A-Lab.
   - 15-iteration regret trajectories (`ElectrolyteOptimizationChart`) for Electrolyte Screening.

---

## 6. Automated Integrity Verification

Automated test gates in `presentation/tests/test_ui_integrity.py`:
- `test_dataset_registry_completeness`: Confirms existence and schema conformity of all 3 authentic benchmarks.
- `test_dataset_registry_zero_fabrications`: Asserts exact match with raw physical files (`ledger_precursor_genome.json`, `screening_quality_diagnostics.json`, `full_policy_matrix.json`).
- `test_campaign_resolution_isolation`: Asserts zero candidate leakage between datasets (`controlled-` vs `PG_` vs `ELEC_`).
- `test_no_synthetic_fabrications_or_fake_constants`: AST scan across all TypeScript/TSX frontend files verifying zero hardcoded multipliers or fake constants.

# AIcoScientist — Scientific Claim & Artifact Audit

**Audit Date:** September 10, 2026  
**Auditor:** AIcoScientist Lead Research-Demo Integration Engineer  
**Repository:** `AndrewSan314/AIcoScientist`  
**Git Commit SHA:** `c2ae7dd0b374283369ad76fb49ce776b8abcbd39`  
**Git Branch:** `integration/multimodal-scientific-engine`  
**Manifest Checksum Status:** 11/11 Source Artifacts SHA-256 Verified (`presentation/data/snapshot_manifest.json`)

---

## 1. Audit Taxonomy & Evidence Classification

To ensure complete transparency during advisor and peer review, every claim, number, and visualization presented in the Mission Control UI is categorized according to this 5-tier evidence taxonomy:

1. **`CONTROLLED_SYNTHETIC`**: Rigorously defined synthetic hypothesis worlds designed to mathematically verify inference guarantees, Monte Carlo bound compliance, and policy ranking stability.
2. **`RETROSPECTIVE_REPLAY`**: Ground-truth historical experimental data where characterization observations are canonical, and the evaluation outcome oracle is strictly firewalled from candidate selection.
3. **`IN_SILICO_APPROXIMATION`**: Computational surrogate models (e.g. ExtraTrees) used to simulate closed-loop optimization dynamics. Explicitly disclaimed as non-physical.
4. **`REFERENCE_ONLY`**: Legacy benchmarks or architectural specifications preserved for comparative completeness.
5. **`NOT_AVAILABLE`**: Modalities or experimental linkages that do not exist in the source archives. Explicitly labeled to prevent hallucinated experimental capability.

---

## 2. Comprehensive Metric & Claim Verification Audit

### Claim 1: "5,333 Evidence Ledger Events Recorded"
- **Claimed Value:** 5,333 events
- **Source Artifact:** `outputs/alab/multimodal/multimodal_validation.json` (`gate_evidence.ledger_event_count: 5333`) and `outputs/alab/multimodal/evidence_ledger.jsonl`
- **Evidence Mode:** `CONTROLLED_SYNTHETIC` & `RETROSPECTIVE_REPLAY`
- **Audit Finding:** **PASS (Strictly Verified).**
- **Presenter Guardrail:** These 5,333 events are **audit and preregistration records** (action scoring, pre-reveal predictions, observation reveals, and belief updates) logging 180 benchmark trajectories and replay campaigns. They are **NOT** 5,333 physical wet-lab experiments. The UI explicitly displays this warning.

---

### Claim 2: "1,035 Real Inorganic Synthesis Samples Evaluated"
- **Claimed Value:** 1,035 samples (1,032 unique targets, 46 precursors, 1,030 XRD, 1,030 Rietveld refinements, 1,009 classified reaction outcomes)
- **Source Artifact:** `outputs/alab/alab_dataset_audit.json` (`candidate_identity.total_candidates: 1035`)
- **Evidence Mode:** `RETROSPECTIVE_REPLAY`
- **Audit Finding:** **PASS (Authentic Open-Science Data).**
- **Provenance:** Sourced from the published A-Lab Precursor Genome (Walters et al., Zenodo DOI: `10.5281/zenodo.21285546`, CC BY 4.0).

---

### Claim 3: "SEM and EDS Modalities are NOT AVAILABLE for Sample Replay"
- **Claimed Value:** 0 / 1,035 candidate-linked SEM/EDS samples (0% availability)
- **Source Artifact:** `outputs/alab/multimodal/modality_inventory.json` (`modalities.SEM.linked_candidate_samples: 0`, `modalities.EDS.linked_candidate_samples: 0`) and `src/domains/alab/multimodal_inventory.py:65`
- **Evidence Mode:** `NOT_AVAILABLE`
- **Audit Finding:** **PASS (Honest Boundary Enforced).**
- **Presenter Guardrail:** While raw archives `sem.zip` and `eds.zip` exist, their file naming reflects precursor groupings rather than sample IDs. AIcoScientist refuses to synthesize fake links; the UI explicitly flags SEM and EDS as `NOT AVAILABLE` for candidate replay.

---

### Claim 4: "180 Full Trajectories Evaluated in Policy Benchmark Matrix"
- **Claimed Value:** 180 multi-step trajectories (6 policies × 6 worlds × 5 seeds)
- **Source Artifact:** `outputs/alab/multimodal/full_policy_matrix.json` (`trajectory_count: 180`)
- **Evidence Mode:** `CONTROLLED_SYNTHETIC`
- **Audit Finding:** **PASS (Exhaustively Verified).**
- **Scope:** 6 worlds (3 clean, 3 stress under misspecified priors), 6 policies (`PURE_HIG`, `HYBRID`, `DISCOVERY_ONLY`, `UNCERTAINTY_ONLY`, `RANDOM_CANDIDATE_FIXED_MODALITY`, `RANDOM_ACTION`), and 5 random seeds (7, 42, 101, 314, 2024).

---

### Claim 5: "48 out of 50 Boolean Validation Gates Passed"
- **Claimed Value:** 48 PASS, 2 FAIL (Total: 50 gates)
- **Source Artifact:** `outputs/alab/multimodal/multimodal_validation.json` (`boolean_gate_pass_count: 48`, `boolean_gate_count: 50`)
- **Evidence Mode:** `CONTROLLED_SYNTHETIC` & `RETROSPECTIVE_REPLAY`
- **Audit Finding:** **PASS (Zero Masking of Failures).**
- **Presenter Guardrail:** 48/50 is **not** a 96% accuracy score. It measures compliance with formal software, schema, bound, and statistical invariants. The 2 failing gates are explicitly displayed in the UI:
  1. `calibration_coverage_gate`: **FAIL** (`REFINEMENT.target_phase_fraction` coverage50 is 95.2% vs nominal 50%, meaning the model is over-dispersed / conservative).
  2. `chemistry_family_generalization_gate`: **FAIL** (Generalization to completely unobserved elemental systems is not established due to dataset singleton grouping).

---

### Claim 6: "Stage-1 Screening Filters 333,333 Electrolyte Formulations with Zero Latent Gap"
- **Claimed Value:** 333,333 candidates screened to WS=200 in 2.535 seconds; 100% latent maximum recovered (0.7886 latent cap, 0.000 screening gap)
- **Source Artifact:** `outputs/electrolyte/benchmark/screening_quality_diagnostics.json` (`search_space_size: 333333`, `working_set_trials.200.screening_runtime_sec: 2.535`, `screening_latent_gap: 0.0`)
- **Evidence Mode:** `CONTROLLED_SYNTHETIC`
- **Audit Finding:** **PASS (Performance Invariant Verified).**

---

### Claim 7: "Negative Result: BoTorch EI Outperforms Hybrid on Single-Property Exploitation"
- **Claimed Value:** BoTorch EI latent regret = 0.0257 ± 0.0364 vs Hybrid latent regret = 0.0788 ± 0.0023; Hybrid cumulative HIG = 1.587 nats vs BoTorch EI = 1.520 nats; Hybrid entropy reduction = 0.9955 nats vs BoTorch EI = 0.5640 nats
- **Source Artifact:** `outputs/electrolyte/benchmark/surrogate_simulation.md` (lines 20–28) and `surrogate_simulation.json`
- **Evidence Mode:** `IN_SILICO_APPROXIMATION`
- **Audit Finding:** **PASS (Non-Linear Trade-off Transparently Reported).**
- **Presenter Guardrail:** We do not claim Hybrid is a universal winner. When the sole objective is scalar property exploitation, standard BoTorch EI achieves lower regret. Hybrid expresses a balanced trade-off, achieving nearly double the scientific entropy reduction (0.995 vs 0.564 nats).

---

### Claim 8: "Monte Carlo HIG Sensitivity Justifies MC=32"
- **Claimed Value:** MC12 vs MC32 evaluated across 60 paired trajectories; rank correlation 0.833–0.934; action sequence agreement 20%–80%
- **Source Artifact:** `outputs/alab/multimodal/hig_trajectory_sensitivity.json` and `outputs/alab/multimodal/multimodal_validation.json:15`
- **Evidence Mode:** `CONTROLLED_SYNTHETIC`
- **Audit Finding:** **PASS (Methodological Justification Verified).**
- **Presenter Guardrail:** A sensitivity test PASS does not prove that MC12 is stable; it proves that MC12 has ranking noise. Based on this finding, MC=32 was adopted for all 180 full matrix runs.

---

### Claim 9: "Exact Mathematical Score Decomposition $S(a) = w_H \widetilde{HIG} + w_D \widetilde{D} - w_C \widetilde{C}$"
- **Claimed Value:** Weighted sum of normalized components matches total action score within $< 10^{-5}$ across all steps; overall score is dimensionless.
- **Source Artifact:** `src/science/multimodal/decision.py:284` and `presentation/data/snapshot.json`
- **Evidence Mode:** `CONTROLLED_SYNTHETIC` & `LIVE_COMPUTED`
- **Audit Finding:** **PASS (Formulation Mathematically Verified).**
- **Presenter Guardrail:** $S(a)$ is a signed dimensionless composite scalar; raw HIG is in nats. Fake multipliers and incorrect units have been audited and removed.

---

### Claim 10: "11/11 Source Artifacts Cryptographically Verified in Manifest"
- **Claimed Value:** 11 offline scientific artifacts verified with SHA-256 checksums matching file system bytes.
- **Source Artifact:** `presentation/data/snapshot_manifest.json` and `presentation/backend/server.py` (`/api/manifest`)
- **Evidence Mode:** `VERIFIED_MANIFEST`
- **Audit Finding:** **PASS (Cryptographic Integrity Verified).**

---

### Claim 11: "Authentic Dataset Registry With Zero Fabricated Numbers or Fake Constants"
- **Claimed Value:** Exactly 3 authentic datasets (`controlled_multimodal_alloy`, `alab_precursor_genome`, `anode_free_electrolyte_screening`) registered with 100% genuine sample counts (12, 1035, 333,333).
- **Source Artifact:** `presentation/data/dataset_registry.json`, `presentation/tests/test_ui_integrity.py` (`test_dataset_registry_completeness`, `test_dataset_registry_zero_fabrications`, `test_campaign_resolution_isolation`)
- **Evidence Mode:** `CONTROLLED_SYNTHETIC`, `RETROSPECTIVE_REPLAY`, `IN_SILICO_APPROXIMATION`
- **Audit Finding:** **PASS (Zero Fabrications Verified via AST Regex Scans and Integration Tests).**
- **Presenter Guardrail:** Leaked candidate pools (`controlled-` vs `PG_` vs `ELEC_`) are prohibited. Electrolyte target observable is cycle-3 capacity (`norm_capacity_3`), never conductivity. A-Lab policy selection is locked to `HYBRID` because its synthesis sequence is a fixed historical lab replay.

---

## 3. Summary of Compliance with User Instructions

| Instruction Requirement | Implementation Status | Verification Document / Screen |
| :--- | :--- | :--- |
| **Preserve Scientific Core** | 100% Preserved. Zero edits to `src/science/`. | `git diff --stat` |
| **No Fabricated Numbers** | 100% Sourced from committed JSON/JSONL artifacts. | `snapshot.json` / `build_snapshot.py` |
| **Decision Cockpit 3-Zone Composition** | Hypothesis Observatory, Candidate Space, Hero Card. | `DecisionCockpitView.tsx` |
| **Preregister → Reveal → Update Loop** | 4 discrete states (A, B, C, D) with animated updates. | `DecisionCockpitView.tsx` |
| **Clear Data Mode Badges** | Persistent badges across all screens. | `ModeBadge.tsx` |
| **A-Lab SEM/EDS Status** | Explicitly marked `NOT AVAILABLE`. | `ALabAtlasView.tsx` |
| **Honest Failing Gates Display** | 48/50 boolean gates with 2 failing gates visible. | `ReadinessView.tsx` |
| **Electrolyte Surrogate Disclaimer** | Prominently displayed with negative result. | `ElectrolyteView.tsx` |
| **Presenter Mode (6–8 min)** | 8 scenes with keyboard navigation and script. | `PresenterMode.tsx` / `ADVISOR_DEMO_SCRIPT.md` |
| **Offline Presentation Capability** | Works without network or model fitting. | `run_mission_control.ps1` |

# AIcoScientist — Source-to-UI Capability & Artifact Mapping

This document provides a strict, byte-for-byte verification mapping connecting every visual component, metric, and interaction in the **Mission Control UI** to its underlying source code file or committed artifact in `AndrewSan314/AIcoScientist`.

---

## 1. Global Navigation & Badges

| UI Element / Feature | Source File / Artifact | Exact Source Key / Hash | Evidence Mode |
| :--- | :--- | :--- | :--- |
| **Integration Commit SHA** | Git repository state | `c2ae7dd0b374283369ad76fb49ce776b8abcbd39` | Verified Git Hash |
| **Active Branch** | Git branch state | `integration/multimodal-scientific-engine` | Verified Git Branch |
| **Cryptographic Manifest** | `presentation/data/snapshot_manifest.json` | 11 source artifacts SHA-256 verified | Verified Manifest |
| **`CONTROLLED_SYNTHETIC` Badge** | `src/science/multimodal/decision.py` | `CLEAN_CORRECTLY_SPECIFIED` / controlled world | Controlled Synthetic |
| **`HISTORICAL_REPLAY` Badge** | `src/science/multimodal/retrospective.py` | `RetrospectiveObservationSet` / A-Lab Replay | Historical Replay |
| **`LIVE_COMPUTED` Badge** | `presentation/backend/server.py` | `MultimodalDecisionEngine.recommend()` | Live Computed |
| **`NOT_AVAILABLE` Badge** | `src/domains/alab/multimodal_inventory.py` | `action_space_supported: False` (SEM/EDS) | Unavailable Modality |

---

## 2. Research Overview (`/overview`)

| UI Metric / Visual Element | Source Artifact Path | Exact Artifact Key / Field |
| :--- | :--- | :--- |
| **"From property optimization..."** | Prompt / Architectural Manifesto | Section 9 Core Statement |
| **1,035 Candidate Samples** | `outputs/alab/alab_dataset_audit.json` | `candidate_identity.total_candidates: 1035` |
| **1,030 XRD / Refinement Linked** | `outputs/alab/alab_dataset_audit.json` | `canonical_xrd_usability.samples_with_scans: 1035`, `is_ledger_canonical_count: 1030` |
| **180 Controlled Trajectories** | `outputs/alab/multimodal/full_policy_matrix.json` | `trajectory_count: 180` (6 policies × 6 worlds × 5 seeds) |
| **5,333 Ledger Audit Events** | `outputs/alab/multimodal/multimodal_validation.json` | `gate_evidence.ledger_event_count: 5333` |
| **333,333 Virtual Electrolyte Pool** | `outputs/electrolyte/benchmark/screening_quality_diagnostics.json` | `search_space_size: 333333` |
| **9-Stage Decision Pipeline** | `src/science/multimodal/decision.py` | `enumerate_actions → recommend → preregister → observe → bayesian_update` |
| **Research Maturity Strip** | `outputs/alab/multimodal/multimodal_validation.json` | `readiness` dictionary |

---

## 3. Decision Cockpit (`/cockpit`)

| UI Component | Source Implementation | Exact Source Field / Method |
| :--- | :--- | :--- |
| **$H_1, H_2, H_3$ Hypotheses** | `outputs/alab/multimodal/hypothesis_definitions.json` | `H1_PHASE_PURITY_LIMITED`, `H2_COMPOSITION_HOMOGENEITY_LIMITED`, `H3_MORPHOLOGY_KINETICS_LIMITED` |
| **Initial Beliefs (33.3% each)** | `src/science/multimodal/decision.py:54` | `self.beliefs = self._normalize(prior_beliefs or {hid: 1.0 ...})` |
| **Flagship Run Selection** | `outputs/alab/multimodal/evidence_ledger.jsonl` | `run_id: "policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID"` |
| **Candidate Action Space** | `src/science/multimodal/decision.py:99` | `MultimodalDecisionEngine.enumerate_actions()` |
| **3D Stark Hologram Sphere** | `presentation/frontend/src/components/StarkHologramSphere.tsx` | 12 candidate nodes + decorative lattice nodes; reduced motion support |
| **Expected HIG (nats)** | `src/science/multimodal/decision.py:141` | `expected_hypothesis_information_gain_diagnostics()` |
| **Discovery Utility** | `src/science/multimodal/decision.py:203` | `_discovery_value(action)` |
| **Normalized Cost** | `src/science/multimodal/decision.py:277` | `cost / max_cost` |
| **Hybrid Policy Weights** | `scripts/run_alab_multimodal_benchmark.py:50` | `w_hig: 0.8, w_discovery: 0.8, w_cost: 2.0` |
| **Net Decision Score $S(a)$** | `src/science/multimodal/decision.py:284` | `w_hig * normalized_hig + w_discovery * normalized_discovery - w_cost * normalized_cost` |
| **Full 24-Action Counterfactual Table** | `presentation/data/snapshot.json` | `step.all_scored_actions` (12 candidates $\times$ 2 modalities) ranked by score gap |
| **Preregistration Event (State B)** | `src/science/multimodal/decision.py:286` | `event: "PREREGISTERED_SELECTED_ACTION"`, `measurement_revealed: False` |
| **Revealed Measurement (State C)** | `outputs/alab/multimodal/evidence_ledger.jsonl` | `event: "MEASUREMENT_REVEALED"`, `observed_measurement` |
| **Posterior Delta (State D)** | `src/science/multimodal/evidence.py:16` | `bayesian_update()`, `posterior_delta = after - before` |
| **Entropy Reduction** | `src/science/multimodal/evidence.py:10` | `entropy(before) - entropy(after)` |

---

## 4. A-Lab Evidence Atlas (`/alab`)

| UI Component | Source File / Artifact | Exact Source Key / Code |
| :--- | :--- | :--- |
| **Precursor Genome Provenance** | `outputs/alab/alab_dataset_audit.json` | `dataset_identity.zenodo: "https://doi.org/10.5281/zenodo.21285546"` |
| **1,035 Physical Sample Catalog** | `data/raw/precursor_genome/ledger_precursor_genome.json` | 1,035 authentic physical samples (`PG_0102` to `PG_1521`) |
| **Sample Search & Quick Selectors** | `presentation/frontend/src/views/ALabAtlasView.tsx` | Interactive search + buttons (`PG_0102`, `PG_0206`, `PG_0309`, `PG_0841`, `PG_1521`) |
| **Modality Linkage Table** | `outputs/alab/multimodal/modality_inventory.json` | `modalities.XRD`, `modalities.REFINEMENT`, `modalities.SEM`, `modalities.EDS` |
| **SEM/EDS "NOT AVAILABLE" Status** | `src/domains/alab/multimodal_inventory.py:65` | `linked_candidate_samples: 0`, `action_space_supported: False` |
| **Real Sample Record (`PG_0309`)** | `data/raw/precursor_genome/ledger_precursor_genome.json` | Target: `Co3B3H9O13`, Precursors: `B(OH)3, Co3O4`, 200°C heating |
| **Canonical XRD Descriptors** | `src/science/multimodal/ontology.py:65` | `XRD.normalized_intensity_std_proxy`, `XRD.spectral_entropy`, `XRD.peak_count_proxy` |
| **Rietveld Observables** | `src/science/multimodal/ontology.py:73` | `REFINEMENT.target_phase_fraction`, `REFINEMENT.rwp_scaled` |
| **Per-Observable Calibration Table** | `outputs/alab/multimodal/per_observable_calibration.json` | `XRD` and `REFINEMENT` MAE, RMSE, NLL, coverage50, coverage90 |
| **`A_LAB_CALIBRATION_PARTIAL`** | `outputs/alab/multimodal/multimodal_validation.json:18` | `readiness.calibration_coverage_status: "A_LAB_CALIBRATION_PARTIAL"` |
| **Over-dispersed Refinement Phase** | `outputs/alab/multimodal/per_observable_calibration.json:78` | `coverage50: 0.95228` (vs target 0.50, error 0.262) |
| **Generalization Holdouts (4 Protocols)** | `src/science/multimodal/retrospective.py:38` | `build_group_holdout_protocols()` (sample, reaction, target, elemental) |
| **`CHEMISTRY_FAMILY_NOT_ESTABLISHED`** | `outputs/alab/multimodal/multimodal_validation.json:14` | `chemistry_generalization_status: "CHEMISTRY_FAMILY_GENERALIZATION_NOT_ESTABLISHED"` |

---

## 5. Policy Benchmark Laboratory (`/benchmarks`)

| UI Component | Source File / Artifact | Exact Source Key / Metric |
| :--- | :--- | :--- |
| **180 Trajectory Design** | `outputs/alab/multimodal/full_policy_matrix.json:4` | 6 policies × 6 worlds × 5 seeds = 180 trajectories |
| **Clean vs Stress Worlds** | `outputs/alab/multimodal/full_policy_matrix.json:5` | `world_types: ["CLEAN_CORRECTLY_SPECIFIED", "STRESS_MISSPECIFIED"]` |
| **6 Policies** | `outputs/alab/multimodal/full_policy_matrix.json:7` | `RANDOM_ACTION`, `RANDOM_CANDIDATE_FIXED_MODALITY`, `UNCERTAINTY_ONLY`, `DISCOVERY_ONLY`, `PURE_HIG`, `HYBRID` |
| **5 Authentic Metric Columns** | `outputs/alab/multimodal/full_policy_matrix.json` | `recovery_rate_MAP`, `mean_final_true_hypothesis_probability`, `mean_entropy_reduction`, `mean_measurement_cost`, `steps_to_posterior_gt_0.8` |
| **HIG Sensitivity (MC12 vs MC32)** | `outputs/alab/multimodal/hig_trajectory_sensitivity.json` | `aggregate_by_world_policy[world:policy].action_sequence_agreement` |
| **HIG Rank Correlation** | `outputs/alab/multimodal/hig_trajectory_sensitivity.json` | `HIG_rank_correlation: 0.833 to 0.934` |
| **MC32 Justification** | `outputs/alab/multimodal/multimodal_validation.json:15` | `HIG_trajectory_sensitivity_status: "USE_32_FOR_FULL_MATRIX"` |

---

## 6. Electrolyte Discovery (`/electrolyte`)

| UI Component | Source File / Artifact | Exact Source Key / Metric |
| :--- | :--- | :--- |
| **333,333 Virtual Candidate Space** | `outputs/electrolyte/benchmark/screening_quality_diagnostics.json:2` | `search_space_size: 333333` |
| **Screening Methodology** | `outputs/electrolyte/benchmark/screening_quality_diagnostics.json:7` | Ridge + RF (40%), Distance (30%), Farthest-point (20%), Random (10%) |
| **Working Set Sizes Tested** | `outputs/electrolyte/benchmark/screening_quality_diagnostics.json:10` | WS=200, WS=500, WS=1000 |
| **Screening Runtime (2.535 sec)** | `outputs/electrolyte/benchmark/screening_quality_diagnostics.json:18` | `screening_runtime_sec: 2.535` for 333k candidates |
| **Latent Maxima Recovery (100%)** | `outputs/electrolyte/benchmark/screening_quality_diagnostics.json:17` | `latent_max_percentile_recovered: 100.0`, `screening_latent_gap: 0.0` |
| **Surrogate Simulator Benchmark** | `outputs/electrolyte/benchmark/surrogate_simulation.json` | ExtraTrees 100 trees, $\sigma=0.02$, seeds 42, 101, 2024 |
| **BoTorch EI Latent Regret (0.0257)** | `outputs/electrolyte/benchmark/surrogate_simulation.md:24` | `BOTORCH_EI_DIRECT` latent regret: 0.0257 ± 0.0364 |
| **Hybrid Latent Regret (0.0788)** | `outputs/electrolyte/benchmark/surrogate_simulation.md:27` | `HYBRID_DEFAULT` latent regret: 0.0788 ± 0.0023 |
| **Hybrid Entropy Red. (0.995 nats)** | `outputs/electrolyte/benchmark/surrogate_simulation.md:27` | `HYBRID_DEFAULT` entropy reduction: 0.9955 nats |
| **Model Coupling Disclaimer** | `outputs/electrolyte/benchmark/surrogate_simulation.md:13` | Frozen univariate surrogate; does not model causal coupling or physical synthesis |

---

## 7. Research Readiness & Software Governance (`/readiness`)

| UI Component | Source File / Artifact | Exact Source Key / Value |
| :--- | :--- | :--- |
| **Boolean Gate Pass Count** | `outputs/alab/multimodal/multimodal_validation.json:102` | `boolean_gate_pass_count: 48` |
| **Boolean Gate Total Count** | `outputs/alab/multimodal/multimodal_validation.json:101` | `boolean_gate_count: 50` |
| **Failing Gate 1: Calibration Coverage** | `outputs/alab/multimodal/multimodal_validation.json:35` | `calibration_coverage_gate: "FAIL"` |
| **Failing Gate 2: Family Generalization** | `outputs/alab/multimodal/multimodal_validation.json:43` | `chemistry_family_generalization_gate: "FAIL"` |
| **Raw HIG Bound Violations (0)** | `outputs/alab/multimodal/multimodal_validation.json:103` | `hig_raw_bound_violation_count: 0` |
| **Software Methodology Status** | `outputs/alab/multimodal/multimodal_validation.json:3` | `CONTROLLED_CLEAN_AND_STRESS_METHODOLOGY_VALIDATED` |
| **Release Status** | `outputs/alab/multimodal/multimodal_validation.json:7` | `release_readiness: "PENDING_EXTERNAL_CI"` |

---

## 8. Backend Integrity & Diagnostic Endpoints

| Endpoint | File Location | Purpose & Guarantee |
| :--- | :--- | :--- |
| `GET /api/manifest` | `presentation/backend/server.py` | Serves SHA-256 hashes of all 11 source artifacts and dataset commit hashes |
| `GET /api/health` | `presentation/backend/server.py` | Verifies presence of snapshot, manifest, and frontend production dist |
| `GET /api/snapshot` | `presentation/backend/server.py` | Delivers verified 3.68 MB offline discovery snapshot |
| `POST /api/diagnostic/smoke-recommend` | `presentation/backend/server.py` | Responsiveness diagnostic test endpoint; returns `mode: "DIAGNOSTIC_SMOKE_TEST"` |

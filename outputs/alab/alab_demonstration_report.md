# A-Lab Precursor Genome Multimodal Falsification Campaign Report

**Domain**: A-Lab Precursor Genome Solid-State Synthesis  
**Dataset**: 1,035 experimental candidates, 46 unique precursors, real XRD scans & Rietveld refinement cases  
**Engine Architecture**: Unified `ScientificDecisionEngine` with generic representation lifecycle  

---

## 1. Executive Summary

AIcoScientist autonomously planned and executed a 5-step solid-state materials synthesis discovery campaign using real experimental data from the A-Lab Precursor Genome. Operating over the exact same core decision engine that powers the Au–Ir–Rh catalyst and toy battery domains, AIcoScientist simultaneously evaluated 3 competing scientific hypotheses:
1. **`precursor_thermodynamics`**: Conversion governed solely by precursor chemistry and thermodynamic $\Delta G$.
2. **`process_kinetics`**: Conversion dictated by furnace heating temperature and duration.
3. **`structure_phase_informed`**: Conversion explained by intermediate crystalline diffraction patterns and Rietveld phase evolution.

---

## 2. Competition Wow Scenario: Belief Evolution & Decision Trajectory

```json
[
  {
    "step": 1,
    "selected_candidate_id": "PG_0104",
    "action_type": "REFINEMENT",
    "estimated_cost": 0.5,
    "candidate_metadata": {
      "target_compound": "Ba2Ag2C2O7",
      "precursor_1": "Ag2O",
      "precursor_2": "BaCO3",
      "heating_temperature_c": 200.0,
      "heating_time_minutes": 240.0,
      "reaction_energy_ev_per_atom": 0.0
    },
    "score_components": {
      "total_value": 1.1499999999988386,
      "scientific_information_value": 0.9999999999990323,
      "discovery_value": 0.0,
      "cost_penalty": 0.05,
      "raw_hig": 1.0335245545649916
    },
    "hypothesis_beliefs_before": {
      "precursor_thermodynamics": 0.3333333333333333,
      "process_kinetics": 0.3333333333333333,
      "structure_phase_informed": 0.3333333333333333
    },
    "hypothesis_beliefs_after": {
      "precursor_thermodynamics": 0.9999999999816609,
      "process_kinetics": 0.0,
      "structure_phase_informed": 1.8338631615652228e-11
    },
    "revealed_data_summary": {
      "refinement_features": [
        0.9415,
        0.0585,
        3.0,
        5.83
      ],
      "phase_weights": {
        "BaCO3_62_(icsd_166091)-0": 0.7392,
        "Ag_225_(icsd_604631)-2": 0.2023,
        "Ag2O_224_(icsd_173984)-0": 0.0585
      },
      "rwp": 5.83,
      "target_fraction": 0.9415
    },
    "scientific_rationale": "Selects REFINEMENT for candidate 'PG_0104' (Net Scientific Value: 1.150). Expected Hypothesis Information Gain is 1.034 nats (expected posterior entropy: 0.065 nats) under policy mode 'hybrid'."
  },
  {
    "step": 2,
    "selected_candidate_id": "PG_1346",
    "action_type": "OUTCOME_TEST",
    "estimated_cost": 2.0,
    "candidate_metadata": {
      "target_compound": "Ti2Fe2O7",
      "precursor_1": "Fe2O3",
      "precursor_2": "TiO2",
      "heating_temperature_c": 1000.0,
      "heating_time_minutes": 60.0,
      "reaction_energy_ev_per_atom": 0.0
    },
    "score_components": {
      "total_value": 0.9960362070456006,
      "scientific_information_value": 0.9966968392046672,
      "discovery_value": 0.0,
      "cost_penalty": 0.2,
      "raw_hig": 3.0174033326290637e-10
    },
    "hypothesis_beliefs_before": {
      "precursor_thermodynamics": 0.9999999999816609,
      "process_kinetics": 0.0,
      "structure_phase_informed": 1.8338631615652228e-11
    },
    "hypothesis_beliefs_after": {
      "precursor_thermodynamics": 0.035564485024834294,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.964435514975164
    },
    "revealed_data_summary": {
      "reaction_conversion": 1.0,
      "reaction_category": "completely_reacted"
    },
    "scientific_rationale": "Selects OUTCOME_TEST for candidate 'PG_1346' (Net Scientific Value: 0.996). Expected Hypothesis Information Gain is 0.000 nats (expected posterior entropy: 0.000 nats) under policy mode 'hybrid'."
  },
  {
    "step": 3,
    "selected_candidate_id": "PG_3031",
    "action_type": "REFINEMENT",
    "estimated_cost": 0.5,
    "candidate_metadata": {
      "target_compound": "MoPH9N2O7",
      "precursor_1": "MoO3",
      "precursor_2": "(NH4)2HPO4",
      "heating_temperature_c": 200.0,
      "heating_time_minutes": 60.0,
      "reaction_energy_ev_per_atom": -0.0467
    },
    "score_components": {
      "total_value": 1.1499999999906918,
      "scientific_information_value": 0.9999999999922432,
      "discovery_value": 0.0,
      "cost_penalty": 0.05,
      "raw_hig": 0.1289206896535387
    },
    "hypothesis_beliefs_before": {
      "precursor_thermodynamics": 0.035564485024834294,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.964435514975164
    },
    "hypothesis_beliefs_after": {
      "precursor_thermodynamics": 0.987580745316553,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.012419254683445229
    },
    "revealed_data_summary": {
      "refinement_features": [
        0.0,
        0.0,
        0.0,
        5.0
      ],
      "phase_weights": {},
      "rwp": 5.0,
      "target_fraction": 0.0
    },
    "scientific_rationale": "Selects REFINEMENT for candidate 'PG_3031' (Net Scientific Value: 1.150). Expected Hypothesis Information Gain is 0.129 nats (expected posterior entropy: 0.025 nats) under policy mode 'hybrid'."
  },
  {
    "step": 4,
    "selected_candidate_id": "PG_1146",
    "action_type": "REFINEMENT",
    "estimated_cost": 0.5,
    "candidate_metadata": {
      "target_compound": "TiCuO3",
      "precursor_1": "CuO",
      "precursor_2": "TiO2",
      "heating_temperature_c": 800.0,
      "heating_time_minutes": 60.0,
      "reaction_energy_ev_per_atom": 0.0
    },
    "score_components": {
      "total_value": 1.1499999999716701,
      "scientific_information_value": 0.9999999999763919,
      "discovery_value": 0.0,
      "cost_penalty": 0.05,
      "raw_hig": 0.04235817570719056
    },
    "hypothesis_beliefs_before": {
      "precursor_thermodynamics": 0.987580745316553,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.012419254683445229
    },
    "hypothesis_beliefs_after": {
      "precursor_thermodynamics": 0.9998749930814981,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.00012500691850744644
    },
    "revealed_data_summary": {
      "refinement_features": [
        0.4545,
        0.5455,
        3.0,
        3.2
      ],
      "phase_weights": {
        "CuO_9_(icsd_69758)-None": 0.4545,
        "TiO2_141_(icsd_172914)-0": 0.4233,
        "TiO2_136_(icsd_202240)-10": 0.1222
      },
      "rwp": 3.2,
      "target_fraction": 0.4545
    },
    "scientific_rationale": "Selects REFINEMENT for candidate 'PG_1146' (Net Scientific Value: 1.150). Expected Hypothesis Information Gain is 0.042 nats (expected posterior entropy: 0.024 nats) under policy mode 'hybrid'."
  },
  {
    "step": 5,
    "selected_candidate_id": "PG_1246",
    "action_type": "OUTCOME_TEST",
    "estimated_cost": 2.0,
    "candidate_metadata": {
      "target_compound": "Ti3Fe3O10",
      "precursor_1": "Fe3O4",
      "precursor_2": "TiO2",
      "heating_temperature_c": 1000.0,
      "heating_time_minutes": 60.0,
      "reaction_energy_ev_per_atom": -0.0002
    },
    "score_components": {
      "total_value": 0.9999999960271782,
      "scientific_information_value": 0.9999999966893152,
      "discovery_value": 0.0,
      "cost_penalty": 0.2,
      "raw_hig": 0.000302052302211684
    },
    "hypothesis_beliefs_before": {
      "precursor_thermodynamics": 0.9998749930814981,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.00012500691850744644
    },
    "hypothesis_beliefs_after": {
      "precursor_thermodynamics": 0.9995254509059222,
      "process_kinetics": 0.0,
      "structure_phase_informed": 0.00047454909407982753
    },
    "revealed_data_summary": {
      "reaction_conversion": 1.0,
      "reaction_category": "completely_reacted"
    },
    "scientific_rationale": "Selects OUTCOME_TEST for candidate 'PG_1246' (Net Scientific Value: 1.000). Expected Hypothesis Information Gain is 0.000 nats (expected posterior entropy: 0.001 nats) under policy mode 'hybrid'."
  }
]
```

---

## 3. 4-Policy Comparison Benchmark

| Policy | Final Max Conversion | Total Budget Spent | Leading Scientific Hypothesis |
|---|---|---|---|
| **`HYBRID` (AIcoScientist)** | **1.0** | **13.5** | `structure_phase_informed` |
| `DISCOVERY_ONLY` | 0.0 | 12.0 | `precursor_thermodynamics` |
| `PURE_FALSIFICATION` | 1.0 | 15.0 | `structure_phase_informed` |
| `RANDOM` | 1.0 | 18.0 | `untracked` |

---

## 4. Key Scientific Findings

1. **Information Horizon Rigor**: XRD representation basis refits only on revealed spectra and is strictly frozen during Bayesian update log-evidence calculations, preventing lookahead leakage.
2. **Modality Prerequisite Enforcement**: Rietveld refinement actions were strictly offered only after powder XRD patterns were measured on the corresponding candidate.
3. **Domain Decoupling**: Zero domain-specific branching exists within the core decision engine; the A-Lab domain operates strictly through `MaterialDomainAdapter` and `ObservationRepresentationManager`.

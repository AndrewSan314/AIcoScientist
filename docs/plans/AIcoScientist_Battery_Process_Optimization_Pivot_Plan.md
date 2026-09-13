# AIcoScientist Pivot Plan
## Multimodal, Stage-Aware Battery Manufacturing Process Optimization

**Repository:** `AndrewSan314/AIcoScientist`  
**Target branch:** `integration/multimodal-scientific-engine`  
**Observed branch HEAD when this plan was prepared:** `bdaf235772ba9f1fd1bcdbfbba0e102456557a04`  
**Planning status:** IMPLEMENTATION BLUEPRINT  
**Primary decision:** make battery-manufacturing **process optimization** the default research direction; preserve the current falsification work only as an auditable legacy/research track.

---

# 0. Executive mandate

The project should pivot from a falsification-first system:

```text
candidate × measurement
    -> expected hypothesis information gain
    -> preregister prediction
    -> reveal measurement
    -> update H1/H2/H3 posterior
```

to a process-optimization system:

```text
raw-material / formulation context
    -> mixing settings
    -> slurry state
    -> coating settings
    -> wet-electrode state
    -> drying settings
    -> dry-electrode state
    -> calendering settings
    -> electrode structure
    -> cell assembly / filling
    -> formation / aging settings
    -> multimodal process measurements
    -> final battery KPIs
    -> recommend improved process controls
```

The new central scientific question is:

> **Given everything observed so far in a battery manufacturing run, which process settings should be chosen next — or which complete process recipe should be executed — to maximize feasible final-cell performance under quality, safety, cost, energy, equipment, and manufacturability constraints?**

This is **not primarily a new-material-discovery project**. Chemistry/material identity may be fixed, stratified, or treated as context. The main decision variables are process controls.

The project's intended contribution should become the combination of:

```text
MULTIMODAL PROCESS STATE
        +
STAGE-AWARE PROCESS-CHAIN MODEL
        +
MISSING-MODALITY ROBUSTNESS
        +
UNCERTAINTY / CALIBRATION
        +
CONSTRAINED MULTI-OBJECTIVE OPTIMIZATION
        +
TRACEABLE CLOSED-LOOP LEARNING
```

Do **not** work on UI until this new core has a real offline benchmark.

---

# 1. Why this direction is scientifically justified

Battery manufacturing is not one regression table. It is a sequence of strongly coupled subprocesses. Process parameters influence intermediate product properties, those intermediate properties propagate downstream, and final cell performance reflects the entire history.

A representative process-modeling paper:

**Changbai Tan et al. — Data-driven battery electrode production process modeling enabled by machine learning**  
Journal of Materials Processing Technology 316 (2023), 117967  
DOI: `10.1016/j.jmatprotec.2023.117967`

The paper models relationships between process parameters and intermediate product properties for electrode-production subprocesses such as mixing, coating, drying, and calendering. This is the architectural idea AIcoScientist should adopt: model the **process chain**, not merely the final endpoint.

The pivot is also consistent with literature on:
- multi-output battery production design;
- physics-assisted battery manufacturing inverse design;
- multimodal battery characterization;
- multimodal formation metrology;
- real-time manufacturing control;
- process-graph explainability.

---

# 2. Literature that must guide implementation

The coding agent must inspect these works before finalizing contracts. Use them as conceptual guidance; do not blindly copy implementations or claims.

## 2.1 Whole process-chain modeling

### Tan et al. (2023)
**Data-driven battery electrode production process modeling enabled by machine learning**  
DOI: `10.1016/j.jmatprotec.2023.117967`

Import these ideas:
- process parameters -> intermediate product properties;
- individual stage models;
- nonlinear interactions;
- explicit process-chain structure;
- reusable stage models.

AIcoScientist implication:
- preserve stage identity;
- intermediate states are first-class prediction targets;
- later-stage predictors may consume earlier-stage predicted/measured state.

---

## 2.2 Real multi-factor formulation/manufacturing optimization

### Drakopoulos et al. (2021)
**Formulation and manufacturing optimization of lithium-ion graphite-based electrodes via machine learning**  
Cell Reports Physical Science 2, 100683  
DOI: `10.1016/j.xcrp.2021.100683`  
Open data DOI: `10.17632/4dh2h3tsf4.1`

Reported experimental variation includes:
- slurry composition;
- mixing protocol;
- electrode coating gap;
- drying temperature;
- coating speed;
- calendering.

Reported outputs include rheological, adhesion, and electrochemical measurements.

The paper describes a difficult small-data / many-factor setting and reports 27 formulation/manufacturing cases, 15 experimental variables in the ML model, 85 cells in training, later test cells, and AI-designed validation cells.

**Use this as the primary real process-optimization benchmark.**

---

## 2.3 Multi-output production design

### Battery production design using multi-output machine learning models
Energy Storage Materials 38 (2021), 93–112  
DOI: `10.1016/j.ensm.2021.03.002`

Import:
- multiple final targets;
- prediction from intermediate product properties;
- inverse derivation of acceptable intermediate-property windows;
- cyber-physical production-system framing.

AIcoScientist implication:
- do not make capacity the only possible target;
- support multi-output model heads and production tolerances.

---

## 2.4 Physics-assisted multi-objective inverse design

### Duquesnoy et al. (2023)
**Machine learning-assisted multi-objective optimization of battery manufacturing from synthetic data generated by physics-based simulations**  
Energy Storage Materials 56, 50–61  
DOI: `10.1016/j.ensm.2022.12.040`

Import:
- physics-generated manufacturing datasets;
- surrogate models;
- multi-objective optimization;
- inverse process design;
- explicit distinction between simulated and physical evidence.

Use this as a methodological precedent for a large **SIMULATED_PHYSICS** stress benchmark.

---

## 2.5 Manufacturing optimization for energy/power

### Duquesnoy et al. (2024)
**Toward high-performance energy and power battery cells with machine learning-based optimization of electrode manufacturing**  
Journal of Power Sources 590, 233674  
DOI: `10.1016/j.jpowsour.2023.233674`

Import:
- process -> microstructure -> performance chain;
- inverse design;
- multiple use-case objectives.

---

## 2.6 Multimodal battery ML

### Bridging multimodal data and battery science with machine learning
Matter 7 (2024), 2011–2032  
DOI: `10.1016/j.matt.2024.04.030`

Import:
- heterogeneous battery data need modality-specific representations;
- images, characterization signals, material descriptors, and condition data carry complementary information;
- multimodal fusion should be evaluated against single-modality baselines.

---

## 2.7 Multimodal formation control

### Qi, Briggs, Marco
**Multimodal metrology for sustainable lithium-ion battery formation: From baseline electrical measurements to intelligent, adaptive process control**  
Renewable and Sustainable Energy Reviews, article 117395  
DOI: `10.1016/j.rser.2026.117395`

The review covers complementary:
- electrical;
- impedance;
- mechanical;
- thermal;
- acoustic;
- spectroscopic

measurements for formation monitoring and adaptive closed-loop control.

Import:
- formation as a dedicated stage;
- multimodal process state;
- sensor fusion;
- eventual closed-loop stage control.

---

## 2.8 Large-scale multimodal workflow and provenance

### Cadiou et al. (2026)
**A Large Scale Multi-Modal Workflow for Battery Characterization: From Concept to Implementation**  
Advanced Energy Materials  
DOI: `10.1002/aenm.71288`

Reported workflow:
- 15 laboratories/facilities;
- 15 techniques;
- 90 samples;
- 19 experiments;
- 75 datasets;
- standardized sample identity and metadata.

Import:
- modality registry;
- sample identity;
- metadata/provenance;
- partial modality availability;
- source-linked multimodal correlation.

Do not make this a P0 dataset dependency if raw data are not fully downloadable.

---

## 2.9 Physics-aware stage surrogate

### A physics-induced surrogate modeling framework for lithium-ion battery electrode manufacturing
Journal of Power Sources 688 (2026), 240416  
DOI: `10.1016/j.jpowsour.2026.240416`

Import:
- slurry/drying/calendering stage models;
- physics-derived engineered features;
- interpretable stage trends;
- balance accuracy and compute.

---

## 2.10 Process graph / path explanation

### Kim et al. (2025)
**Decoding industrial-scale battery manufacturing process through integration of causal graphs into explainable artificial intelligence**  
Engineering Applications of Artificial Intelligence 159, 111657  
DOI: `10.1016/j.engappai.2025.111657`

Import:
- explicit process dependency graph;
- accumulated influence through unit processes;
- path-based feature attribution;
- Shapley-Flow-style explanations.

Important wording rule:
- use `process dependency graph`, `domain-informed process graph`, or `causal-prior graph`;
- do not claim causal identification unless interventions/causal assumptions actually justify it.

---

## 2.11 Continuous process control

### Multi-criteria and real-time control of continuous battery cell production steps using deep learning
Advances in Industrial and Manufacturing Engineering 6, 100108  
DOI: `10.1016/j.aime.2022.100108`

Import:
- temporal process telemetry;
- controller-oriented architecture;
- quality + cost + sustainability objectives;
- future real-time/receding-horizon mode.

---

# 3. Current repository: what should be reused

Do not rebuild AIcoScientist from zero.

The current repo already contains a large amount of reusable infrastructure.

## 3.1 Keep: generic optimizer abstraction

Current production optimizer has:
- `OptimizerBackend`;
- official BoTorch production path;
- strict strategy resolution;
- random;
- greedy/posterior mean;
- GP-UCB;
- Expected Improvement;
- Noisy Expected Improvement;
- Thompson sampling;
- finite candidate pools;
- candidate identity;
- content fingerprints;
- deterministic seeds.

Reuse these.

Do not reintroduce bespoke generic BO math into production.

---

## 3.2 Keep: ledger, identity, provenance, resume

Preserve:
- experiment/run IDs;
- authoritative ledger philosophy;
- data fingerprints;
- candidate/process recipe fingerprints;
- snapshot compatibility checks;
- deterministic resume;
- artifact provenance.

Adapt semantics from "candidate experiment" to:
- process run;
- process stage;
- process recipe;
- process-control proposal.

---

## 3.3 Keep/revive: original battery process pipeline

The older repository design already includes:
- manufacturing process features;
- SEM images/features;
- EDX;
- electrochemical outputs;
- master dataset;
- RF baseline;
- GP surrogate;
- UCB recommendation;
- human feedback.

This is closer to the new direction than the current falsification-first presentation path.

Audit all these components and classify:
- KEEP;
- MODIFY;
- LEGACY;
- DELETE ONLY AFTER replacement is verified.

---

## 3.4 Keep: SEM/image processing

Reuse valid existing:
- image ingest;
- threshold/segmentation fallback;
- morphology metrics;
- crack metrics;
- mask/overlay artifacts.

In the new system these become an **IMAGE modality encoder/feature adapter**, not a standalone flagship.

---

## 3.5 Keep: benchmark discipline

Reuse:
- multi-seed runs;
- best-so-far curves;
- regret;
- calibration tests;
- source manifests;
- artifact hashing;
- fail-closed data validation.

---

# 4. Current repository: structural gaps that must be closed

The repo is **not yet complete for the boss's new direction**.

Major missing capabilities:

1. genuine learned multimodal fusion;
2. stage-aware manufacturing state model;
3. typed process-stage contracts;
4. explicit controls vs observations distinction;
5. stage-based information horizon;
6. missing-modality contract;
7. production-ready multi-output model interface;
8. genuine constrained optimization;
9. genuine multi-objective BoTorch path;
10. process recipe search-space abstraction;
11. manufacturability/failure model;
12. grouped/batch/OOD manufacturing evaluation;
13. stage-wise contextual optimization;
14. unified battery-process dataset adapters.

Important current limitation:

`OptimizationObjective` already has fields such as `constraints`, `secondary_targets`, and `weights`, but the frozen production optimizer specification currently rejects true constrained and multi-objective optimization.

For this pivot, that limitation must be deliberately lifted using **official BoTorch/Ax primitives**.

---

# 5. Retire falsification from the default product path

Do not delete historical research.

Retire from the default architecture:

```text
H1/H2/H3 as a mandatory top-level abstraction
expected HIG as the primary objective
candidate × measurement-modality as the universal action
preregistration/falsification as the defining loop
controlled hypothesis worlds as the flagship benchmark
A-Lab hypothesis replay as the primary product story
```

Preserve for reproducibility under either:

```text
src/legacy/falsification/
docs/legacy/falsification/
```

or keep files in place but mark them clearly:

```text
LEGACY_RESEARCH_TRACK
NOT_DEFAULT_BATTERY_PROCESS_PATH
```

Required rule:

```text
DEFAULT_PRODUCT_MODE = BATTERY_PROCESS_OPTIMIZATION
```

The new process optimizer must have **zero dependency on H1/H2/H3/HIG**.

---

# 6. Canonical battery manufacturing stage model

The schema must support the full chain even if any individual dataset covers only part of it.

## S0 — Material / formulation context

Possible context/controls:
- chemistry;
- active material;
- binder;
- conductive additive;
- solvent;
- material fractions;
- solids content;
- particle properties;
- supplier/batch.

## S1 — Mixing

Controls:
- mixing sequence;
- speed;
- time;
- temperature;
- shear/energy input.

Observations:
- rheology/viscosity;
- dispersion;
- mixer power/torque;
- temperature time series.

## S2 — Coating

Controls:
- coating gap;
- line speed;
- flow rate;
- coating ratio.

Observations:
- wet thickness;
- mass loading;
- optical image;
- equipment telemetry.

## S3 — Drying

Controls:
- zone temperatures;
- airflow;
- line speed/residence time.

Observations:
- residual solvent;
- temperature profile;
- dry thickness;
- morphology;
- optical/thermal measurements.

## S4 — Calendering

Controls:
- roll pressure/gap;
- roll temperature;
- roll speed.

Observations:
- porosity;
- density;
- thickness;
- tensile/adhesion;
- ultrasound;
- SEM/EDS/XCT if available.

## S5 — Slitting / cutting / assembly

Controls:
- dimensions;
- stack/winding settings;
- welding/alignment.

Observations:
- geometry;
- optical QC;
- equipment telemetry.

## S6 — Electrolyte filling / wetting

Controls:
- electrolyte amount;
- vacuum/pressure;
- wetting time;
- temperature.

Observations:
- mass;
- acoustic/ultrasound;
- impedance if available.

## S7 — Formation / aging

Controls:
- current profile;
- voltage cutoff;
- temperature;
- rest times;
- formation duration.

Observations:
- voltage/current curves;
- impedance;
- thermal;
- acoustic;
- pressure/mechanical;
- gas/spectroscopy if available.

## S8 — Final characterization

Targets:
- capacity;
- capacity retention;
- coulombic efficiency;
- impedance / DCIR / ASI;
- power;
- energy density;
- rate capability;
- cycle life;
- defect/yield;
- process cost;
- process energy.

---

# 7. New core mathematical model

At stage `t`:

\[
z_t = F_t(z_{t-1}, u_t, m_t, c)
\]

where:

- `z_{t-1}` = inherited upstream process state;
- `u_t` = controllable settings at current stage;
- `m_t` = available multimodal observations;
- `c` = fixed context: chemistry, equipment, batch, environment.

Final KPI distribution:

\[
p(y \mid z_T) = G(z_T)
\]

or equivalently:

\[
p(y \mid u_{1:T}, m_{1:T}, c)
\]

For recipe-level optimization:

\[
u_{1:T}^{*}
=
\arg\max_{u_{1:T}}
\mathcal{U}\left(p(y \mid u_{1:T},c)\right)
\]

subject to:

\[
g_j(u_{1:T}, z_{1:T}) \le 0.
\]

For multiple objectives, prefer a Pareto formulation over arbitrary fixed scalarization.

---

# 8. New process data contracts

Create a process-centric package.

Suggested:

```text
src/process/
    contracts.py
    stages.py
    modalities.py
    information_horizon.py
    feature_store.py
    validation.py
    coordinator.py
    fusion/
    models/
    optimization/
    explain/
```

## 8.1 BatteryProcessRun

```python
@dataclass(frozen=True)
class BatteryProcessRun:
    run_id: str
    cell_id: str | None
    batch_id: str | None
    chemistry_id: str
    equipment_context: dict[str, Any]
    environment_context: dict[str, Any]
    stages: list["StageRecord"]
    final_kpis: dict[str, "MeasurementValue"]
    provenance: "ProvenanceRecord"
```

---

## 8.2 StageRecord

```python
@dataclass
class StageRecord:
    stage_id: str
    stage_type: ProcessStage
    sequence_index: int
    controls: dict[str, ParameterValue]
    intermediate_properties: dict[str, MeasurementValue]
    modalities: list["ModalityObservation"]
    upstream_stage_id: str | None
    timestamps: StageTimeRange | None
    quality_flags: list[str]
    provenance: ProvenanceRecord
```

Controls and observations must never be conflated.

---

## 8.3 ModalityObservation

```python
@dataclass
class ModalityObservation:
    modality_id: str
    modality_type: str
    observed_at_stage: ProcessStage
    source_path: str | None
    values: Any
    units: str | None
    shape: tuple[int, ...] | None
    missing_reason: str | None
    quality_score: float | None
    provenance: ProvenanceRecord
```

Allowed modality families:

```text
PROCESS_TABULAR
MACHINE_TIME_SERIES
THERMAL_TIME_SERIES
OPTICAL_IMAGE
SEM_IMAGE
EDS_MAP
XCT_VOLUME
ULTRASOUND_SIGNAL
ULTRASOUND_SPECTRUM
EIS_SPECTRUM
XRD_SPECTRUM
FORMATION_CURVE
CYCLING_CURVE
SCALAR_METROLOGY
ENVIRONMENT_METADATA
```

Missing data must be explicit.

Never replace missing modalities with synthetic values and label them observed.

---

# 9. Information-horizon firewall

This is P0.

When making a stage decision, the model must only consume variables genuinely available before that decision.

Example:

When selecting `CALENDERING` pressure:

Allowed:
- formulation;
- mixing controls/measurements;
- coating controls/measurements;
- drying controls/measurements.

Forbidden:
- post-calendering porosity;
- future ultrasound recorded after calendering;
- cell capacity;
- 50-cycle retention.

Create:

```python
InformationHorizon(stage=ProcessStage.CALENDERING)
```

It should return allowed:
- controls;
- intermediate properties;
- modalities;
- targets.

All dataset adapters and model pipelines must pass the horizon validator.

This generalizes the current repo's no-lookahead philosophy into process optimization.

---

# 10. Multimodal model architecture

Do not begin with one giant transformer.

Battery manufacturing public datasets are small and heterogeneous. Build a strong baseline ladder.

## 10.1 Tabular/process encoder

Inputs:
- scalar process controls;
- intermediate physical properties;
- chemistry/equipment categorical context.

Baselines:
- ExtraTrees;
- RandomForest;
- CatBoost/XGBoost if dependency policy permits.

Neural only after baseline:
- compact MLP.

---

## 10.2 Time-series encoder

For:
- mixer power;
- line telemetry;
- drying temperature;
- formation current/voltage;
- equipment signals.

Baseline:
- physical/statistical descriptors.

Advanced:
- 1D CNN;
- TCN;
- compact Transformer.

---

## 10.3 Spectrum encoder

For:
- ultrasound FFT;
- EIS;
- XRD/spectroscopy when available.

Baseline:
- peak/band/summary features.

Advanced:
- 1D CNN;
- compact spectral Transformer.

---

## 10.4 Image encoder

For:
- SEM;
- EDS maps;
- optical coating inspection;
- XCT slice/volume.

First baseline:
- reuse current morphology/crack/porosity feature extraction.

Advanced:
- pretrained image encoder;
- freeze backbone initially;
- fine-tune only with enough data.

Do not train a huge vision model from scratch on tens of cells.

---

## 10.5 Electrochemical curve encoder

For:
- voltage/current formation curves;
- capacity-vs-cycle;
- impedance evolution.

Baseline:
- physically meaningful engineered descriptors.

Advanced:
- temporal encoder.

---

# 11. Missing-aware multimodal fusion

## Phase A — gated masked fusion

Each modality encoder outputs:

```text
z_modality ∈ R^d
```

Then:

```text
available modality tokens
      + modality masks
      + stage embedding
          ↓
learned gated pooling / attention pooling
          ↓
stage multimodal state
```

This is the P1 target because:
- it works with partial modalities;
- it is much easier to validate on small data;
- modality contribution can be inspected.

## Phase B — cross-attention / set fusion

Only after Phase A beats baselines.

Use a set of available modality tokens so missing modalities do not require fixed zero-filled slots.

Required:
- missing-modality mask;
- no future-stage token leakage;
- modality ablation;
- robustness under structured modality dropout.

---

# 12. Stage-aware prediction model

A flat all-feature predictor is required as a baseline but should not be the final architecture.

Concept:

```python
z = encode_context(run)

for stage in process_order:
    stage_obs = fuse(stage.modalities)
    z = stage_transition(
        upstream_state=z,
        stage_type=stage.stage_type,
        controls=stage.controls,
        observations=stage_obs,
    )
```

Final heads:

```text
capacity
retention
impedance
energy
power
yield
```

Intermediate heads:

```text
viscosity
wet_thickness
dry_thickness
porosity
density
adhesion
```

Multi-task supervision is preferred when labels exist.

---

# 13. Process dependency graph

Create a declarative graph file:

```text
config/process_graphs/li_ion_electrode.yaml
```

Example:

```yaml
nodes:
  - formulation
  - mixing_controls
  - slurry_rheology
  - coating_controls
  - wet_thickness
  - drying_controls
  - dry_morphology
  - calendering_controls
  - porosity
  - density
  - formation_controls
  - impedance
  - capacity

edges:
  - [formulation, slurry_rheology]
  - [mixing_controls, slurry_rheology]
  - [slurry_rheology, coating_controls]
  - [coating_controls, wet_thickness]
  - [drying_controls, dry_morphology]
  - [calendering_controls, porosity]
  - [porosity, impedance]
  - [impedance, capacity]
```

Purpose:
- legal information flow;
- stage interpretability;
- path attribution;
- architecture prior.

Do not call it proven causal unless data support causal identification.

---

# 14. Multi-output prediction

The target API must support multiple outputs with per-target availability masks.

Minimum supported target family:

```text
capacity
capacity_retention
coulombic_efficiency
impedance
porosity
density
thickness
manufacturing_success
```

Optional:
- energy density;
- power;
- cycle life;
- adhesion;
- tensile strength;
- defect probability;
- manufacturing energy/cost.

Do not require every dataset to expose every target.

---

# 15. Uncertainty and calibration

Process optimization needs calibrated uncertainty.

Implement selectable uncertainty methods:

```text
GP posterior
bootstrap/tree ensemble
deep ensemble
conformal prediction
heteroscedastic regression where justified
```

Minimum calibration report:

```text
50% coverage
80% coverage
90% coverage
95% coverage
interval width
calibration error
NLL / CRPS if probabilistic output exists
```

Evaluation must include:
- IID split;
- process-condition group split;
- batch holdout;
- OOD process settings.

---

# 16. Optimizer pivot

The new action is **not** universally `candidate × measurement modality`.

## Mode 1 — complete process recipe optimization

Implement first.

```python
ProcessRecipe = {
    "mixing_protocol": ...,
    "mixing_speed": ...,
    "mixing_time": ...,
    "coating_gap": ...,
    "coating_speed": ...,
    "drying_temperature": ...,
    "calendering_pressure": ...,
    "calendering_temperature": ...,
}
```

Useful for retrospective finite-pool replay.

---

## Mode 2 — stage-wise contextual optimization

Implement second.

At stage `t`:

```text
context = upstream state + available observations
decision = next-stage controls
```

\[
u_t^* = \arg\max_{u_t} \alpha(u_t \mid z_{t-1},m_{1:t-1})
\]

This directly matches the boss's requested "optimize each stage".

---

## Mode 3 — receding-horizon process control

Advanced.

```text
current state
    -> predict downstream trajectory
    -> optimize final KPI
    -> execute only next control
    -> measure new state
    -> re-optimize
```

Do not implement as P0.

---

# 17. Multi-objective constrained optimization

This is a major new requirement.

## Maximize examples

```text
capacity
retention
energy density
power
yield
```

## Minimize examples

```text
impedance
defect probability
cost
energy consumption
formation time
```

## Constraints

```text
component fractions valid
equipment parameter bounds
temperature/pressure limits
thickness window
porosity window
minimum adhesion
manufacturability probability >= threshold
```

Use official BoTorch primitives.

Recommended eventual production support:
- constrained EI / log EI;
- noisy EI;
- qNEHVI / qLogNEHVI or current supported official equivalent;
- hypervolume;
- Pareto set;
- reference-point validation.

Do not revive retired custom/fake TuRBO implementations.

---

# 18. Manufacturability / feasibility model

Build:

\[
P(\mathrm{manufacturing\ success}\mid \mathrm{recipe},\mathrm{context})
\]

Failure labels may include:
- mixing failure;
- coating defect;
- out-of-spec electrode;
- adhesion failure;
- assembly rejection;
- missing downstream test because process failed.

Failed experiments are useful data.

Use constrained acquisition, or conceptually:

\[
\alpha_\mathrm{feasible}(x)
=
\alpha(x)\,
P(\mathrm{feasible}\mid x)
\]

when mathematically appropriate.

---

# 19. Dataset strategy: do not search for one mythical "perfect" dataset

Do not require one public dataset containing simultaneously:
- the whole battery manufacturing chain;
- hundreds of variables;
- images;
- spectra;
- equipment time series;
- large N;
- final cycling;
- complete traceability.

That public dataset is unlikely to exist as one clean open package.

Instead create:

# Battery Process Stress Suite (BPSS)

Each task remains scientifically independent.

Do **not** concatenate unrelated datasets/chemistries into one fake master experimental table.

---

# 20. BPSS-1 — Graphite Process-15

## Source

**Graphite-based electrodes for Li-ion batteries: Formulation and manufacturing process optimization via machine learning**  
Mendeley Data DOI: `10.17632/4dh2h3tsf4.1`  
License reported by Mendeley: CC BY 4.0

Related paper:
Drakopoulos et al. 2021.

## Role

Primary **many-factor real manufacturing** benchmark.

Reported process dimensions include:
- slurry formulation;
- mixing protocol;
- coating gap;
- drying temperature;
- coating speed;
- calendering.

## Required tasks

1. audit exact downloaded files;
2. map all source columns;
3. identify exact process controls and outcomes;
4. preserve the original later-test split if recoverable;
5. endpoint prediction;
6. process-recipe offline optimization;
7. group-by-protocol holdout;
8. OOD process-factor holdout;
9. uncertainty calibration;
10. feature/stage attribution.

## Why this is hard

Small N relative to process dimensionality forces:
- careful regularization;
- grouped validation;
- uncertainty;
- no giant neural model.

This is precisely the stress regime needed.

---

# 21. BPSS-2 — Warwick NMC622 Pilot Multimodal

## Source

**Data of Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes and Lithium-ion Cells at Pilot-Plant Scale Battery Manufacturing**  
Mendeley Data DOI: `10.17632/wwhm2frfmy.1`  
License reported by Mendeley: CC0 1.0

Related Data in Brief article:
DOI `10.1016/j.dib.2023.109798`

Related process study:
DOI `10.1016/j.jpowsour.2023.233091`

## Reported design

18 unique conditions:
- calender roll temperature: 85 / 120 / 145 °C;
- porosity: 30 / 35 / 40%;
- mass loading: 120 / 180 g/m².

54 half cells, three replicates per condition.

Reported response families include:
- thickness;
- coating weight;
- tensile strength;
- density;
- capacities at multiple C-rates;
- volumetric and gravimetric capacities;
- area-specific impedance across SoC;
- 50-cycle performance;
- degradation;
- coulombic efficiency.

The repository also contains physical/electrochemical raw files and should be audited for equipment and image assets.

## Role

Primary **real multimodal pilot-line benchmark**.

## Required benchmark modes

```text
process tabular only
physical metrology only
image only, when actual images are source-linked
electrochemical early features only
process + physical
process + image
process + signal
all available modalities
```

Then:
- missing-modality ablation;
- grouped-by-process-condition holdout.

Do not split replicates from one DOE condition across train/test for the main generalization benchmark.

---

# 22. BPSS-3 — Warwick Ultrasonic Inline QC

## Source

**Ultrasonic Signal Dataset for Battery Electrode Thickness Prediction**  
Mendeley dataset family DOI: `10.17632/c62yn37d9h`

At the time of planning, public versions include:
- time-domain signal release;
- frequency-domain releases.

Use the latest verified source version in the adapter manifest.

Reported signals:
- graphite anode;
- NMC622 cathode;
- before/after calendering;
- raw or FFT ultrasonic response;
- process metadata;
- thickness target.

## Role

Force the new model to handle **high-dimensional non-tabular signals**.

## Tasks

- signal-only thickness regression;
- process-only regression;
- process + ultrasound fusion;
- before/after-calender state;
- anode -> cathode / cathode -> anode domain shift if scientifically meaningful;
- missing-signal stress;
- spectral encoder benchmark.

---

# 23. BPSS-4 — Na-ion High-Throughput Process Chain

## Source

**Formation and cycling data for Na-ion batteries from high-throughput synthesis, coating, and assembly**  
Zenodo DOI: `10.5281/zenodo.7981011`

Reported archive:
- 660 MB;
- batch synthesis;
- screen printing;
- robotic assembly;
- formation/cycling.

## Role

Process-chain data engineering / lineage stress.

Use for:
- heterogeneous-file ingest;
- batch/run identity;
- synthesis -> coating -> assembly -> cycling lineage;
- cross-stage feature availability.

Do not assume it exposes dozens of independently varied process factors until audited.

---

# 24. BPSS-5 — ARTISTIC Physics Stress

## Source

**ARTISTIC Project — Online Calculator — Battery Manufacturing Simulations**  
Zenodo record `5956128`

The workflow explicitly runs:
- slurry;
- drying;
- calendering

in sequence and generates 3D electrode/manufacturing simulation outputs.

## Role

Large, explicitly simulated, process-space stress.

Use for:
- many candidate recipes;
- stage transition checks;
- multi-objective optimization;
- OOD stress;
- scaling;
- digital-twin-compatible interfaces.

Every output must carry:

```text
evidence_kind = SIMULATED_PHYSICS
```

Never call it physical validation.

---

# 25. Battery process dataset registry

Create:

```text
data/battery_process_registry.yaml
```

Each entry must contain:

```yaml
id:
display_name:
chemistry:
evidence_kind:
source_doi:
dataset_doi:
version:
license:
source_paths:
source_hashes:
sample_count:
run_count:
process_stages:
modalities:
control_variables:
intermediate_targets:
final_targets:
grouping_keys:
known_missingness:
known_leakage_fields:
recommended_splits:
optimization_capable:
multimodal_capable:
limitations:
```

No unsupported counts or fields.

---

# 26. Dataset adapter protocol

Create:

```python
class BatteryProcessDatasetAdapter(Protocol):
    def metadata(self) -> BatteryDatasetMetadata: ...
    def load_runs(self) -> list[BatteryProcessRun]: ...
    def validate(self) -> DatasetValidationReport: ...
    def build_training_view(self, task: ProcessPredictionTask) -> ProcessTrainingFrame: ...
    def build_optimization_space(self, task: ProcessOptimizationTask) -> ProcessSearchSpace: ...
```

Initial adapters:

```text
DrakopoulosGraphiteAdapter
WarwickNMC622Adapter
WarwickUltrasoundAdapter
NaIonHTEAdapter
ArtisticSimulationAdapter
```

For every adapter create:

```text
DATASET_AUDIT.md
SOURCE_TO_SCHEMA.md
manifest.json
```

---

# 27. Baseline ladder

Multimodal claims require strong baselines.

## B0
Random / historical best for optimization.

## B1
Flat tabular ExtraTrees / RF.

## B2
Strong gradient-boosted tabular model when allowed.

## B3
Stage-aware scalar model without raw multimodal encoders.

## B4
Each modality separately.

## B5
Naive feature concatenation.

## M1
Missing-aware gated multimodal fusion.

## M2
Cross-attention multimodal fusion.

## M3
Stage-aware multimodal transition model.

Do not claim M3 is superior unless held-out/grouped results support it.

---

# 28. Evaluation split policy

Random row split is not enough.

Required split types where source permits:

```text
IID_RANDOM
GROUP_BY_PROCESS_RECIPE
GROUP_BY_DOE_CONDITION
GROUP_BY_BATCH
LEAVE_ONE_PROCESS_SETTING_OUT
LATER_RUN_HOLDOUT
OOD_FACTOR_EXTREME
CROSS_ELECTRODE_TYPE
```

Main scientific claims should use grouped/OOD splits.

Replicates of the same process condition must remain in the same group in the main benchmark.

---

# 29. Multimodal stress tests

Create a controlled stress layer without altering source raw data.

## Missing modality rates

```text
10%
25%
50%
75%
```

## Structured modality loss

```text
no SEM
no EDS
no ultrasound
no machine telemetry
no early electrochem
process controls only
```

## Stage dropout

Hide intermediate measurements from one stage.

## OOD controls

Hold out high/low values of:
- temperature;
- mass loading;
- porosity;
- line speed;
- calender setting.

## Noise

Only create a derived `SIMULATED_STRESS` copy with recorded seed/noise model.

## Nuisance dimensions

Only in synthetic stress:
- irrelevant variables;
- correlated nuisance variables;
- missingness patterns.

Never present nuisance-expanded data as physical source data.

---

# 30. Metrics

## Prediction

```text
MAE
RMSE
R²
Spearman
NLL
CRPS
```

where applicable.

## Calibration

```text
50% / 80% / 90% / 95% empirical coverage
mean interval width
calibration error
```

## Multimodal value

```text
gain over best single modality
gain over naive concatenation
modality ablation delta
stage ablation delta
performance under missing modality
```

## Optimization

```text
best-so-far
simple regret
normalized regret
threshold-hit rate
steps-to-threshold conditional on crossing
constraint-violation rate
Pareto hypervolume
number of process trials
```

## Manufacturing

When source supports:

```text
yield
failure probability
out-of-spec rate
energy
cost
```

---

# 31. Explainability

The system must answer:

1. Which manufacturing stage matters most for this target?
2. Which controllable setting matters?
3. Which intermediate property carries the association downstream?
4. Which modality adds predictive value?
5. Why is the proposed recipe preferred?
6. Which constraints are active?

Implement:
- global feature importance;
- local SHAP;
- modality ablation;
- stage ablation;
- process-path attribution.

Do not use causal language without a justified causal design.

---

# 32. New ProcessOptimizationCoordinator

Create:

```text
ProcessOptimizationCoordinator
```

Responsibilities:

```text
load source process state
validate provenance
validate information horizon
assemble available modalities
encode/fuse process state
predict intermediate/final KPIs
estimate uncertainty
construct feasible process-control space
call OptimizerBackend
rank process recipes or stage controls
persist proposal
reveal source outcome in offline replay
ingest new observation
resume deterministically
```

Do not force `MultimodalDecisionEngine` to perform this if that engine remains coupled to hypothesis HIG.

Reuse lower-level infrastructure through composition.

---

# 33. New optimization contracts

Suggested:

```python
@dataclass
class ObjectiveSpec:
    target: str
    sense: Literal["maximize", "minimize"]
    units: str | None = None

@dataclass
class ConstraintSpec:
    name: str
    type: str
    threshold: float | tuple[float, float]
    hard: bool = True

@dataclass
class ProcessOptimizationObjective:
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec]
    reference_point: list[float] | None = None
    feasibility_threshold: float | None = None
```

Proposal:

```python
@dataclass
class ProcessControlProposal:
    proposal_id: str
    stage: ProcessStage | None
    controls: dict[str, Any]
    predicted_outputs: dict[str, Prediction]
    feasibility_probability: float | None
    acquisition_value: float | None
    pareto_rank: int | None
    model_version: str
    data_fingerprint: str
```

---

# 34. Mixed process search spaces

Support:

```text
continuous:
  temperature
  pressure
  speed
  time
  flow

integer:
  number of cycles
  stage count

categorical:
  protocol
  equipment mode
  material/binder choice when treated as context/control

compositional:
  fractions with sum constraint

conditional:
  parameters active only for selected process mode
```

P0 may discretize authentic ranges to a finite candidate pool because current AIcoScientist has strong finite-pool semantics.

Later add proper mixed-variable Ax/BoTorch support.

---

# 35. Offline closed-loop replay

For a historical finite set of source process recipes:

```text
take initial observed subset
    -> train model
    -> optimizer selects one unobserved recorded recipe
    -> reveal source outcome only after selection
    -> update training set
    -> repeat
```

Critical:
- hidden outcomes may exist in storage but cannot enter model/scoring before reveal;
- proposal must be chosen only using current horizon;
- exact source recipe ID retained.

Compare:
- random;
- greedy;
- GP-UCB;
- EI;
- NEI;
- Thompson.

Metrics:
- best-so-far;
- regret;
- threshold success;
- constraint violations;
- calibration.

---

# 36. Recipe-level optimization milestone

The first meaningful non-UI demo must run via CLI.

Example:

```bash
python scripts/run_process_demo.py \
  --dataset drakopoulos_graphite \
  --task recipe_optimization \
  --target <verified_capacity_target> \
  --strategy noisy_expected_improvement \
  --seed 42
```

Output:

```text
dataset source
process stages
controllable factors
target
initial training observations
model validation
proposed recipe
predicted target ± uncertainty
feasibility
source outcome after reveal
best-so-far
regret
```

Do not invent a target column name. The adapter must expose verified source names.

---

# 37. Multimodal benchmark milestone

Example:

```bash
python scripts/run_multimodal_process_benchmark.py \
  --dataset warwick_nmc622 \
  --target <verified_target> \
  --split group_by_process_condition
```

Output:

```text
tabular baseline
physical-metrology baseline
image/signal-only baseline
naive concatenation
gated multimodal fusion
stage-aware multimodal model
modality ablation
missing-modality stress
calibration
```

---

# 38. Multi-objective inverse-design milestone

Example input:

```yaml
objectives:
  - target: capacity
    sense: maximize
  - target: impedance
    sense: minimize

constraints:
  - manufacturability_probability >= 0.90
  - porosity within source/equipment range
  - temperature within equipment range
```

Output:
- Pareto recipes;
- predicted objectives;
- uncertainty;
- feasibility;
- closest historical process recipes;
- changed process stages;
- provenance.

---

# 39. Recommended repository structure

```text
src/
  process/
    contracts.py
    stages.py
    modalities.py
    information_horizon.py
    validation.py
    coordinator.py

    fusion/
      tabular.py
      timeseries.py
      spectrum.py
      image.py
      electrochem.py
      gated_fusion.py
      cross_attention.py

    models/
      flat_baseline.py
      multioutput.py
      stage_transition.py
      process_graph.py
      uncertainty.py

    optimization/
      process_objective.py
      constraints.py
      process_space.py
      proposal.py

    explain/
      stage_importance.py
      modality_ablation.py
      path_attribution.py

  datasets/
    battery_process/
      drakopoulos_graphite.py
      warwick_nmc622.py
      warwick_ultrasound.py
      naion_hte.py
      artistic.py

  optimization/
    # retain generic official BoTorch backend

  legacy/
    falsification/
      # only if migration is safe
```

Do not perform mechanical file moves that break imports.

A compatibility layer is acceptable.

---

# 40. Implementation roadmap

## Phase 0 — repository pivot audit

Create:

```text
docs/pivot/CURRENT_REPO_REUSE_AUDIT.md
docs/pivot/BATTERY_PROCESS_OPTIMIZATION_ARCHITECTURE.md
docs/pivot/LEGACY_FALSIFICATION_BOUNDARY.md
docs/pivot/DATASET_RESEARCH.md
```

For every current module:

```text
component
file
current role
KEEP / MODIFY / LEGACY
new role
migration risk
tests
```

No core behavior changes yet.

---

## Phase 1 — process contracts + horizon

Implement:
- stage enum;
- process run;
- stage record;
- modality observation;
- measurements/controls;
- provenance;
- information horizon;
- serialization.

Acceptance:
- all BPSS datasets can be represented without fake fields;
- future-stage leakage test passes.

---

## Phase 2 — authentic dataset adapters

Order:

1. Drakopoulos graphite.
2. Warwick NMC622.
3. Warwick ultrasound.
4. Na-ion HTE.
5. ARTISTIC.

For each:
- verify DOI/version/license;
- download raw data;
- hash raw files;
- preserve immutable raw data;
- create adapter;
- create dataset audit;
- create split groups;
- create deterministic processed cache.

Do not commit huge raw data unless repository policy explicitly allows it.

---

## Phase 3 — strong scalar/tabular baselines

Implement:
- ExtraTrees;
- RF;
- optional boosting;
- simple GP for low-dimensional tasks;
- uncertainty baselines;
- grouped CV.

Produce baseline report before multimodal work.

---

## Phase 4 — modality encoders

Implement:
- tabular;
- signal/spectrum;
- image adapter;
- electrochemical curve;
- machine time series.

Every encoder gets its own test and single-modality benchmark.

---

## Phase 5 — gated multimodal fusion

Implement:
- common embedding dimension;
- modality token type;
- modality masks;
- gated attention/pooling;
- missing-modality training.

Run:
- unimodal;
- naive concatenate;
- gated fusion.

No superiority claim until results support it.

---

## Phase 6 — stage-aware process model

Implement:
- stage embeddings;
- upstream state;
- stage transition;
- intermediate heads;
- final heads;
- process graph prior.

Compare:
- flat model;
- stage-aware scalar;
- stage-aware multimodal.

---

## Phase 7 — constrained / multi-objective optimizer

Extend current official BoTorch backend.

Implement:
- constraints;
- objective vectors;
- reference point;
- Pareto/frontier;
- hypervolume;
- official qNEHVI/qLogNEHVI equivalent;
- fail-closed unsupported semantics.

Update optimizer architecture freeze document.

---

## Phase 8 — offline process optimization replay

Implement no-lookahead replay for BPSS datasets where meaningful.

Run >= several seeds.

Do not repeatedly rerun until a favorable result appears.

Preserve negative results.

---

## Phase 9 — physics scale stress

Use ARTISTIC or another verified physics-based manufacturing simulator.

Generate explicit simulation manifests.

Stress:
- process dimensionality;
- candidate pool size;
- OOD controls;
- missing intermediate observations;
- multi-objective frontier.

---

## Phase 10 — integrated BPSS benchmark

Create:

```text
scripts/run_battery_process_benchmark.py
```

Output:

```text
outputs/process_benchmark/
  manifest.json
  dataset_audits/
  prediction/
  calibration/
  multimodal_ablations/
  missing_modality/
  stage_ablations/
  optimization/
  stress/
  figures/
  PROCESS_BENCHMARK_REPORT.md
```

---

# 41. Required tests

Create at minimum:

```text
tests/process/test_contracts.py
tests/process/test_stage_order.py
tests/process/test_information_horizon.py
tests/process/test_dataset_adapters.py
tests/process/test_provenance.py
tests/process/test_grouped_splits.py
tests/process/test_modality_encoders.py
tests/process/test_missing_modality.py
tests/process/test_fusion.py
tests/process/test_stage_model.py
tests/process/test_multioutput.py
tests/process/test_uncertainty.py
tests/process/test_constraints.py
tests/process/test_multiobjective_backend.py
tests/process/test_process_space.py
tests/process/test_offline_replay_no_leakage.py
tests/process/test_resume.py
tests/process/test_bpss.py
```

P0 invariants:
- exact process-run identity;
- no future-stage feature leakage;
- no hidden final target leakage;
- no fake missing-modality values;
- no invalid process recipe proposal;
- no silent optimizer fallback;
- exact source hash/provenance;
- no cross-dataset row contamination.

---

# 42. Scientific-claim rules

Allowed only when supported by benchmark evidence:

```text
"The stage-aware model improves grouped-condition prediction by X."
"Multimodal fusion improves held-out prediction versus the best unimodal baseline by X."
"Constrained BO reaches the target within N revealed historical process trials."
"Prediction intervals achieve X% coverage on grouped holdout."
"Physics-simulated benchmark scales to N process recipes."
```

Not allowed without evidence:

```text
"Optimizes industrial battery manufacturing."
"Proven causal process model."
"Autonomously controls a factory."
"Best manufacturing recipe."
"Multimodal always beats unimodal."
"Generalizes across battery chemistries."
```

Use exact evidence labels:

```text
PHYSICAL_HISTORICAL
PILOT_LINE_HISTORICAL
SIMULATED_PHYSICS
SIMULATED_STRESS
OFFLINE_REPLAY
LIVE_PROCESS
```

---

# 43. Data provenance

External data layout:

```text
data/external/<dataset>/<version>/raw/
```

Processed:

```text
data/processed/<dataset>/<adapter_version>/
```

Every manifest:

```text
source URL
DOI
version
license
download timestamp
raw SHA-256
adapter git SHA
schema version
processing parameters
processed hashes
```

Never overwrite raw files.

---

# 44. Performance strategy for small real datasets

A "strong multimodal model" does **not** mean a huge model.

Do:
- use physical/statistical signal features as baselines;
- freeze pretrained image encoder initially;
- regularize aggressively;
- low-dimensional fusion;
- grouped CV;
- nested tuning if necessary;
- calibrate uncertainty;
- use transfer/pretraining only with auditable source.

Do not:
- train millions of parameters on 54 cells and report random-split R²;
- put replicates from one process condition in both train and test;
- call random row split "generalization".

---

# 45. Recommended stress-test hierarchy

The hardest credible test should combine:

```text
REAL SMALL-N MULTI-FACTOR PROCESS DATA
        +
REAL HIGH-DIMENSIONAL SIGNAL/IMAGE DATA
        +
MISSING MODALITIES
        +
GROUPED/OOD SPLITS
        +
UNCERTAINTY CALIBRATION
        +
OFFLINE OPTIMIZATION REPLAY
        +
LARGE PHYSICS-SIMULATED PROCESS SPACE
```

This is stronger than pretending a single large flat table proves factory-level capability.

---

# 46. Initial implementation target

If resources are limited, start with the electrode process rather than the entire cell factory:

```text
FORMULATION
    ↓
MIXING
    ↓
COATING
    ↓
DRYING
    ↓
CALENDERING
    ↓
ELECTRODE CHARACTERIZATION
    ↓
CELL CAPACITY / RETENTION / IMPEDANCE
```

Recommended first suite:

1. **Drakopoulos Graphite Process-15**
   - difficult real multi-factor process optimization.

2. **Warwick NMC622 Pilot Multimodal**
   - real pilot-line process + physical/electrochemical data.

3. **Warwick Ultrasonic**
   - high-dimensional inline sensing.

4. **ARTISTIC**
   - large stage-aware physics simulation stress.

Na-ion HTE then adds end-to-end workflow/lineage diversity.

---

# 47. New README positioning

Primary description:

> **AIcoScientist is a multimodal battery-manufacturing process optimization platform. It links controllable manufacturing settings and heterogeneous measurements across production stages to intermediate product states and final cell performance, then uses uncertainty-aware constrained optimization to recommend improved process conditions.**

Secondary:

> The system is validated first through source-backed offline replay, multimodal process benchmarks, and physics-simulated stress tests. Live factory control is an extension point, not a current claim.

---

# 48. Required coding-agent workflow

Before implementation:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -10 --oneline
```

The agent must **not** assume the planning SHA remains HEAD.

Then generate:

```text
docs/pivot/CURRENT_REPO_REUSE_AUDIT.md
```

Do not implement everything in one commit.

Recommended commit sequence:

```text
1. docs(pivot): define battery process optimization architecture
2. feat(process): add battery manufacturing stage contracts
3. feat(process): enforce stage information horizon
4. feat(data): add graphite process dataset adapter
5. feat(data): add Warwick NMC622 adapter
6. feat(data): add ultrasonic process-signal adapter
7. feat(model): add process prediction baselines
8. feat(model): add missing-aware multimodal fusion
9. feat(model): add stage-aware process model
10. feat(opt): add process recipe optimization
11. feat(opt): add constrained multiobjective BoTorch support
12. feat(eval): add Battery Process Stress Suite
13. docs(pivot): freeze process architecture and benchmark evidence
```

Run relevant tests after every phase.

---

# 49. Definition of done

The pivot is complete only when:

- process is represented as ordered manufacturing stages;
- controls and observations are distinct;
- multimodal observations are typed;
- missing modalities remain explicit;
- information horizon prevents future leakage;
- at least two authentic real battery-process datasets are integrated;
- at least one real dataset stresses multiple manufacturing factors;
- at least one real dataset stresses non-tabular signals/images;
- a strong tabular baseline exists;
- multimodal fusion exists;
- stage-aware model exists;
- grouped/OOD evaluation exists;
- uncertainty is evaluated/calibrated;
- offline optimization replay exists;
- process proposals contain exact settings;
- provenance is auditable;
- true multi-objective/constraint support is either implemented or explicitly reported as incomplete;
- falsification/HIG is not required anywhere in the default battery process path;
- simulated data are never presented as physical evidence.

---

# 50. Final deliverables required from the agent

At the end of each implementation milestone provide:

## Architecture
- changed package tree;
- reuse map;
- legacy boundary.

## Dataset evidence
For every dataset:
- DOI;
- version;
- license;
- exact file inventory;
- exact run/sample count;
- stage coverage;
- modalities;
- process factors;
- targets;
- grouping keys;
- missingness;
- raw hashes.

## Model
- baseline;
- encoder;
- fusion;
- stage model;
- uncertainty method;
- number of trainable parameters.

## Optimization
- objective;
- constraints;
- search-space contract;
- optimizer strategy;
- replay protocol;
- no-leakage validation.

## Evaluation
- grouped prediction metrics;
- calibration;
- modality ablation;
- missing-modality stress;
- stage ablation;
- OOD stress;
- optimization regret;
- Pareto/hypervolume if multi-objective.

## Limitations
Explicitly list:
- sample size;
- unavailable modalities;
- simulated evidence;
- lack of live production control;
- chemistry generalization status.

## Reproducibility
- exact command;
- exact commit SHA;
- environment;
- tests;
- artifact manifest.

---

# 51. Final agent mandate

Do **not** work on presentation UI yet.

Do **not** spend implementation time making the old falsification screens fit the new direction.

Do **not** invent a giant "multimodal battery dataset" by joining unrelated public datasets.

Do **not** invent missing process settings or measurements.

Do **not** claim causal process effects from correlational data.

Do **not** replace official BoTorch acquisition math with custom heuristics.

The immediate technical milestone is:

```text
AUTHENTIC BATTERY PROCESS DATA
        ↓
TYPED STAGE-AWARE REPRESENTATION
        ↓
MULTIMODAL ENCODERS
        ↓
MISSING-AWARE FUSION
        ↓
INTERMEDIATE + FINAL KPI PREDICTION
        ↓
CALIBRATED UNCERTAINTY
        ↓
CONSTRAINED PROCESS-RECIPE OPTIMIZATION
        ↓
OFFLINE NO-LOOKAHEAD REPLAY
        ↓
REAL + PHYSICS-SIMULATED STRESS SUITE
```

The new project's strongest claim should eventually be:

> **AIcoScientist learns and optimizes battery manufacturing as a multimodal sequential process, rather than treating battery performance as a one-shot material-ranking problem.**

# Generic Scientific Closed-Loop Experimentation Architecture

This document describes the domain-generic closed-loop experimentation and reasoning framework in **AIcoScientist**.

---

## 1. Scientific Overview & Information Horizons

The architecture models the core scientific inquiry loop across materials, chemistry, and battery engineering systems:

```
+-------------------------------------------------------------------------------+
|                       INFORMATION HORIZONS & BOUNDARIES                       |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [Horizon 1 & 2: Pre-Experiment Knowns & Controllable Process Variables]     |
|      Process Features (X): Temperatures, Pressures, Ratios, Times             |
|                                     │                                         |
|                                     ▼                                         |
|  [Stage A Surrogate: P(C | X)]                                                |
|      Multi-channel Independent GPs predicting physical structure /            |
|      spectroscopic characterization channels: C = (z_1, z_2, ...)             |
|                                     │                                         |
|                                     ▼                                         |
|  [Horizon 3: Post-Experiment Structure / Characterization (C)]               |
|      Physical Characterization: SEM, XRD, Raman, EDX, Porosity, Phases        |
|      *At proposal time, C is unknown -> integrated via Monte Carlo draws*     |
|                                     │                                         |
|                                     ▼                                         |
|  [Stage B Surrogate: P(Y | X, C)]                                             |
|      Mechanistic Performance GP mapping (Process X + Structure C) -> Y        |
|                                     │                                         |
|                                     ▼                                         |
|  [Horizon 4: Downstream Performance Outcomes (Y)]                             |
|      Functional properties: Lifetime, Capacity, Retention, Conductivity       |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### Search Space & Variable Semantics
- **Supported Variables**: Continuous numeric (`ContinuousVariable`) and discrete numeric (`DiscreteVariable`).
- **Categorical Variables**: Categorical variables are supported once explicitly mapped to deterministic integer or ordinal encodings. Arbitrary raw string categorical GP kernels are not currently implemented.

---

## 2. Mathematical Foundation: Uncertainty Propagation

At candidate proposal time, physical characterization has not occurred. To evaluate expected downstream performance $\mathbb{E}[Y \mid x]$ and total epistemic uncertainty $\text{Var}[Y \mid x]$, the system performs **Monte Carlo integration with the Law of Total Variance**:

1. **Draw Characterization Fantasies**:
   $$C^{(k)} \sim \mathcal{N}\left(\mu_C(x), \text{diag}(\sigma_C^2(x))\right), \quad k = 1, \dots, K$$

2. **Evaluate Stage B Surrogate**:
   $$\left(\mu_Y^{(k)}, \sigma_Y^{(k)}\right) = \text{StageB}\left(x, C^{(k)}\right)$$

3. **Decompose Total Epistemic & Predictive Variance**:
   $$\mathbb{E}[Y \mid x] = \frac{1}{K}\sum_{k=1}^K \mu_Y^{(k)}$$

   $$\text{Var}_{\text{latent}}[Y \mid x] = \underbrace{\frac{1}{K}\sum_{k=1}^K \left(\sigma_Y^{(k)}\right)^2}_{\text{performance model variance}} + \underbrace{\frac{1}{K-1}\sum_{k=1}^K \left(\mu_Y^{(k)} - \mathbb{E}[Y \mid x]\right)^2}_{\text{characterization propagation variance}}$$

   $$\text{Var}_{\text{predictive}}[Y \mid x] = \text{Var}_{\text{latent}}[Y \mid x] + \sigma^2_{\text{obs\_noise}}$$

---

## 3. Transactional Proposal Semantics & Pending Resume

To guarantee safe and deterministic physical execution:

1. **Transactional Proposal Generation**:
   - The coordinator creates a deep clone of active `OptimizerState`.
   - The optimizer evaluates the next candidate using this prospective clone (`step += 1`, prospective trust region state).
   - Candidate parameters, static pre-experiment context, direct model predictions, two-stage predictions, and `ScientificRationale` are evaluated and validated against `DatasetSpec` boundaries.
   - If any validation check fails (e.g. context conflict, missing process variable, non-finite value), the active optimizer state remains **100% unchanged** (no step mutation or leaked trust region adjustments).
   - Upon successful validation, the proposal is committed to the ledger, the optimizer snapshot is hash-anchored, and active state is promoted.

2. **Crash & Resume with a Pending Proposal**:
   - If the process terminates after a proposal is committed but before experimental observations are recorded, `resume_from_ledger()` restores the pending experiment and its exact `ExperimentProposal` metadata (`candidate_id`, `design_variables`, `acquisition_score`, `reason_code`, `step`).
   - When the pending experiment's measurements eventually arrive, `observe()` receives the exact original proposal semantics, ensuring uninterrupted mathematical convergence.

---

## 4. Asynchronous Measurements & Component Training Horizons

Physical experimentation frequently yields partial and asynchronous data:

1. **Partial Measurements**:
   - **Characterization**: Channels can arrive incrementally (e.g. $z_1$ first, $z_2$ later). The experiment is not marked complete until all required channels (`spec.post_experiment_characterization`) are available.
   - **Performance**: Primary target (`spec.target_column`) is required for completion. Optional secondary performance targets can arrive incrementally without prematurely completing the record.
   - **Duplicate Detection**: Submitting different values for an existing channel raises `DuplicateMeasurementError` unless `allow_measurement_revision=True` is explicitly passed for audited revisions.

2. **Component-Specific Training Horizons**:
   - **Stage A ($X \to C$)**: Trains on all records with valid process features and characterization, including `CHARACTERIZED` records whose performance is still pending.
   - **Direct Baseline ($X \to Y$)**: Trains on all records with valid process features and primary target, including `PERFORMANCE_MEASURED` records whose characterization is pending.
   - **Stage B ($X, C \to Y$)**: Trains independently per target $y_j$ on records having valid process features, all required characterization inputs, and $y_j$.
   - **Optimizer**: Evaluates completed experiments meeting all completion criteria.

---

## 5. Tamper-Evident Ledger & Provenance

All experimental events and transitions are persisted to an append-only SQLite ledger (`ExperimentLedger`):

- **Tamper-Evident SHA-256 Hash Chaining**:
  $$H_i = \text{SHA256}\left(H_{i-1} \,\|\, \text{canonical\_json}\left(\{\text{experiment\_id}, \text{event\_type}, \text{created\_at}, \text{payload}\}\right)\right)$$
- **Snapshot Hash Anchoring**: Optimizer snapshots are committed directly into the hash chain as `OPTIMIZER_STATE_SNAPSHOT` events.
- **Head & Event-Count Verification**: `verify_integrity()` checks SHA-256 chain continuity from genesis $H_0$, confirms `event_count` and `head_hash` against stored metadata, and cross-checks projection summary tables against event replay.
- **Scientific Model Provenance**: `ScientificModelProvenance` tracks deterministic SHA-256 fingerprints across datasets (preserving declared feature ordering), specs, and component-specific training experiment IDs, dynamically refreshing `model_run_id` upon every asynchronous model refit.

---

## 6. Scientific Rationale & Learning Value

Every proposal generates a structured `ScientificRationale` answering 6 core scientific questions:

1. **WHAT SHOULD WE TEST?**: Candidate parameters and coordinates relative to nearest observed experiment.
2. **PREDICTED PERFORMANCE**: Expected target mean and latent uncertainty $\pm \sigma$, plus model disagreement $|\mu_{\text{direct}} - \mu_{\text{two\_stage}}|$.
3. **EXPECTED STRUCTURE / CHARACTERIZATION**: Stage A predictions per characterization channel with latent error bars.
4. **WHY THIS EXPERIMENT?**: Optimizer strategy and acquisition score.
5. **WHAT WILL WE LEARN?**: Quantified `expected_learning_value` score bounded in $[0, 1]$ combining normalized uncertainty, spatial novelty, and model disagreement ratio $r / (1+r)$.
6. **SCIENTIFIC CAVEATS**: Explicit warnings that characterization values are surrogate estimates rather than physically verified values.

---

## 7. Running the Synthetic Demo

```bash
# Run a 5-step autonomous closed-loop experimentation cycle on synthetic data
python -m src.science.cli demo --seed 42 --steps 5
```

Artifacts generated in `outputs/scientific_demo/`:
- `proposal_history.jsonl`: Step-by-step proposals and rendered rationales.
- `model_report.json`: 4-way honest evaluation metrics (Direct, Stage A, Diagnostic B upper-bound, End-to-End Two-Stage).
- `run_provenance.json`: Environment, library versions, and git fingerprints.

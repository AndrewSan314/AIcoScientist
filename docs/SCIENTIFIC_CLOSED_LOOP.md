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

## 3. Tamper-Evident Experiment Ledger

All experimental events and transitions are persisted to an append-only SQLite ledger (`ExperimentLedger`) secured by **tamper-evident SHA-256 event hash chaining that detects modification or deletion of hashed historical events while the expected chain head/event count remains available**:

- **Genesis Hash**: $H_0 = \text{"0"} \times 64$
- **Canonical Event Envelope**: Full structured dictionary including `experiment_id`, `event_type`, `created_at`, `payload`.
- **Event Hash**: $H_i = \text{SHA256}(H_{i-1} \,\|\, \text{canonical\_json}(\text{envelope}_i))$
- **Integrity Verification**: `verify_integrity()` recomputes the entire chain from genesis and cross-checks the projection tables (`experiments.current_stage`) against replayed event history.
- **Deterministic Snapshots**: The ledger stores exact `OPTIMIZER_STATE_SNAPSHOT` records including `OptimizerState` step counter, current best, and `TuRBOTrustRegion` state (`TrustRegionState`) for exact, bitwise-consistent resume.

---

## 4. Scientific Lifecycle & Stage Transitions

```
                               ┌──────────────┐
                               │   PROPOSED   │
                               └──────┬───────┘
                                      │ (Dispatched / Executed)
                               ┌──────▼───────┐
                               │   EXECUTED   │
                               └───┬──────┬───┘
               (Char arrives first)│      │ (Perf arrives first)
                   ┌───────────────┘      └───────────────┐
                   ▼                                      ▼
           ┌──────────────┐                       ┌──────────────────────┐
           │ CHARACTERIZED│                       │ PERFORMANCE_MEASURED │
           └──────┬───────┘                       └──────────┬───────────┘
                  │ (Perf arrives)                           │ (Char arrives)
                  └───────────────┐      ┌───────────────────┘
                                  ▼      ▼
                               ┌──────────────┐
                               │  COMPLETED   │
                               └──────────────┘
```

- **Symmetric Asynchronous Lifecycle**: Supports either characterization arriving first (`CHARACTERIZED`) or performance arriving first (`PERFORMANCE_MEASURED`).
- **Direct Datasets**: Datasets without characterization targets transition directly from `EXECUTED` to `COMPLETED` when performance is recorded.
- **Prospective Pre-Commit Validation**: Transitions are validated on an in-memory clone against `DatasetSpec` before executing any SQL commit.
- **Experimental Failures**: A failed synthesis is recorded with `stage=FAILED` and a descriptive `failure_reason`. It is never converted into fabricated zero or worst-case target values.
- **Monotonic ID Sequencing**: Sequence counters increment monotonically even across `FAILED` or `CANCELLED` experiments to prevent experiment ID collisions.

---

## 5. Structured Deterministic Scientific Rationale

Every proposed experiment generates a structured `ScientificRationale` answering 6 core scientific questions:

1. **WHAT SHOULD WE TEST?**: Candidate parameters and coordinates relative to nearest observed experiment.
2. **PREDICTED PERFORMANCE**: Expected target mean and latent uncertainty $\pm \sigma$, plus model disagreement $|\mu_{\text{direct}} - \mu_{\text{two\_stage}}|$.
3. **EXPECTED STRUCTURE / CHARACTERIZATION**: Stage A predictions per characterization channel with latent error bars.
4. **WHY THIS EXPERIMENT?**: Optimizer strategy (e.g. Expected Improvement, TuRBO-NEI, GP-UCB) and acquisition score.
5. **WHAT WILL WE LEARN?**: Quantified `expected_learning_value` score bounded strictly in $[0, 1]$ combining normalized uncertainty, spatial novelty, and model disagreement ratio $r / (1+r)$.
6. **SCIENTIFIC CAVEATS**: Explicit warnings that characterization values are surrogate estimates rather than physically verified values.

---

## 6. Running the Synthetic Demo

```bash
# Run a 5-step autonomous closed-loop experimentation cycle on synthetic data
python -m src.science.cli demo --seed 42 --steps 5
```

Artifacts generated in `outputs/scientific_demo/`:
- `proposal_history.jsonl`: Step-by-step proposals and rendered rationales.
- `model_report.json`: 4-way honest evaluation metrics (Direct, Stage A, Diagnostic B upper-bound, End-to-End Two-Stage).
- `run_provenance.json`: Environment, library versions, and git fingerprints.

# Architectural Decision Record: Migration to BoTorch Optimization Backend

## Status: ACCEPTED & IMPLEMENTED

## Context & Motivation

AIcoScientist is a scientific closed-loop discovery framework specializing in:
- Scientific closed-loop reasoning and experimental state tracking
- Multimodal materials, synthesis process, and physical characterization modeling (e.g. XRD, SEM, electrochemistry, battery cycling)
- Experimental ledger, crash recovery, and cryptographic provenance audit trails
- Experiment lifecycle governance and information horizon firewalls
- Scientific replay and cross-library transfer benchmarks

Previously, AIcoScientist maintained custom implementations of Gaussian process surrogates, analytical acquisition functions (EI, UCB), Monte Carlo acquisition functions (MC-NEI), trust region algorithms (TuRBO), and heuristic adaptive controllers.

### Architectural Decision

AIcoScientist **must not reinvent generic Bayesian optimization infrastructure**. We are not building a generic numerical optimization library or BoTorch competitor.

From this point forward:
- Generic Bayesian optimization mathematics, surrogate fitting, transformation pipelines, and acquisition functions are strictly delegated to mature external libraries (**BoTorch** / **GPyTorch** / **PyTorch**).
- AIcoScientist acts as the scientific reasoning and domain layer on top of interchangeable optimization backends.

---

## Layered Architecture & Separation of Concerns

```
+-------------------------------------------------------------------------------+
|                      AIcoScientist Scientific Domain Layer                    |
|  - ScientificClosedLoopCoordinator, TwoStageScientificModel, DirectBaseline  |
|  - ExperimentLedger, Cryptographic Provenance, Information Horizon Firewall  |
|  - Real Dataset Adapters (Fe-Co-Ni, Au-Ir-Rh, Si-MXene, Attia 2020)           |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                       Optimization Protocol Abstraction                       |
|  - OptimizerBackend (Protocol in src/optimization/backend.py)                |
|  - OptimizationObjective (Single/Multi-objective, Sense, Bounds, Constraints)  |
|  - CandidateProposal & ExperimentProposal (ID, Mean, Std, Acquisition Score)  |
|  - FiniteCandidatePool (Discrete pool management & Observed filtering)       |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                         BoTorch Production Backend                            |
|  - BoTorchBackend (SingleTaskGP, Normalize, Standardize, fit_gpytorch_mll)    |
|  - Standardized Acquisition Mappings:                                         |
|      * Random / Space-filling                                                 |
|      * Greedy / Posterior Mean                                                |
|      * Upper Confidence Bound (qUCB / Analytic UCB)                           |
|      * Expected Improvement (qExpectedImprovement / LogEI)                    |
|      * Noisy Expected Improvement (qNoisyExpectedImprovement / LogNEI)        |
|      * Thompson Sampling (Joint Multivariate Posterior Sampling)             |
+-------------------------------------------------------------------------------+
```

---

## Finite Pool Optimization Semantics

In physical materials discovery and high-throughput experimental synthesis, candidates often belong to a **discrete, pre-synthesized, or combinatorial candidate library** (e.g. 921 Fe-Co-Ni combinatorial library points, 966 Au-Ir-Rh compositions, or finite synthesized battery protocols).

### Strict Candidate Identity Preservation
- `BoTorchBackend.propose()` evaluates acquisition functions directly over the finite unobserved candidate points:
  $$x^* = \arg\max_{x \in \mathcal{X}_{\text{unobserved}}} \alpha(x \mid \mathcal{D})$$
- Proposing points via continuous relaxation followed by nearest-neighbor projection is strictly prohibited, as it can cause duplicate selections, out-of-domain drift, or invalid physical compositions.
- Proposals retain their canonical string `candidate_id` and all associated design variable metadata throughout the closed loop.

---

## Scale Invariance Guarantees

In physical sciences, objective measurements may be expressed in varied physical units (e.g. coercivity in Oe or kOe, exchange currents in A/cm² or mA/cm², cycle life in cycles or log-cycles).

### Mathematical Property
For any positive affine transformation of the target variable:
$$y' = a \cdot y + b \quad (a > 0, b \in \mathbb{R})$$
the ranking of candidate proposals under invariant acquisition strategies (greedy, UCB, EI, NEI) remains identical:
$$\arg\max_{x} \alpha(x \mid \mathcal{D}') = \arg\max_{x} \alpha(x \mid \mathcal{D})$$

### Implementation
- `BoTorchBackend` uses `Standardize(m=1)` for target outcomes and `Normalize(d=d)` for input features.
- Model fitting optimizes Marginal Log Likelihood in the normalized space.
- Predictions and acquisition scores are scaled consistently, guaranteeing no dataset-dependent heuristics or artificial threshold tuning.

---

## Scientific Governance & Provenance Integrity

1. **Information Horizon Firewall**:
   - The optimization backend has zero access to ground-truth oracles, unqueried characterization spectra (XRD, SEM, EDS), or future trajectory states.
   - Only observed experimental records stored in `ExperimentLedger` are provided as `observations`.

2. **Full Replay & Crash Recovery**:
   - Every proposal records its acquisition strategy, predicted mean, epistemic uncertainty, acquisition value, and random seed.
   - Complete scientific state can be resumed from the SQLite ledger or replayed deterministically from initial conditions.

---

## Legacy Archival

All custom Gaussian Process plumbing, bespoke analytical acquisition functions, custom TuRBO trust region implementations, and heuristic controllers have been retired from active production and archived in:
`src/legacy/native_optimizer/`

Production pipelines, science coordinators, and active benchmarks import exclusively from `src.optimization`.

# Optimizer Infrastructure Freeze & Specification Reference

**Status**: FROZEN & ENFORCED  
**Version**: 1.0.0  
**Effective Date**: August 2026  
**Primary Backend**: BoTorch (`src.optimization.botorch_backend.BoTorchBackend`)

---

## 1. Executive Architecture Mandate

AIcoScientist is a scientific reasoning, materials/process characterization, and closed-loop experiment lifecycle platform. **AIcoScientist does not build, reinvent, or maintain generic Bayesian optimization infrastructure.**

All generic Bayesian optimization surrogate fitting, transforms, and acquisition calculations are delegated to **BoTorch** (PyTorch / GPyTorch ecosystem).

```
+-----------------------------------------------------------------------------------+
|                        Scientific Closed-Loop Reasoning Layer                     |
|  - ScientificClosedLoopCoordinator, TwoStageScientificModel, DirectBaseline       |
|  - ExperimentLedger (Authoritative SQLite Ledger & Provenance Chain)             |
|  - Information Horizon Firewall (Zero Lookahead / Zero Oracle Contamination)      |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                          Optimization Protocol Abstraction                        |
|  - OptimizerBackend (Protocol: propose, name, version)                            |
|  - OptimizationObjective (Target name, Minimize/Maximize sense, Constraints)      |
|  - FiniteCandidatePool (Strict string candidate_id identity preservation)        |
|  - CandidateProposal & ProposalMetadata                                           |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                           Production BoTorch Backend                              |
|  - SingleTaskGP with Standardize(m=1) and Normalize(d=d)                         |
|  - Exact MLL Optimization via fit_gpytorch_mll                                    |
|  - Standard Acquisition Functions (Analytic / Monte Carlo / Joint Thompson)       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Standardized Acquisition Strategies & Aliases

The backend supports exactly 6 standard strategies. Any unmapped or non-standard strategy is rejected immediately with `UnsupportedStrategyError`.

| Canonical Strategy Name | Accepted Aliases | Mathematical Implementation | BoTorch Primitive |
| :--- | :--- | :--- | :--- |
| `random` | `random_sampling`, `uniform` | Uniform random selection over unobserved candidates | Pseudo-random discrete sample |
| `greedy` | `posterior_mean`, `exploit` | Posterior mean prediction $\mu(x)$ | `model.posterior(X).mean` |
| `gp_ucb` | `ucb`, `gp_ucb_1`, `gp_ucb_2` | Upper Confidence Bound: $\mu(x) + \sqrt{\beta} \cdot \sigma(x)$ | Analytic UCB / Posterior variance |
| `expected_improvement` | `ei`, `log_ei` | Analytic Expected Improvement over incumbent | `LogExpectedImprovement` / `ExpectedImprovement` |
| `noisy_expected_improvement` | `nei`, `q_nei`, `log_nei` | Monte Carlo / Log Noisy Expected Improvement | `qLogNoisyExpectedImprovement` / `qNoisyExpectedImprovement` |
| `thompson` | `thompson_sampling`, `ts` | Joint discrete multivariate Gaussian posterior sample | `model.posterior(X).rsample()` |

### Prohibited & Retired Heuristics
- **No Fake TuRBO Labels**: `turbo_nei` and `turbo_ei` aliases are completely removed. If passed to `BoTorchBackend`, `UnsupportedStrategyError` is raised.
- **No Silent Fallback**: If `noisy_expected_improvement` fails during evaluation, it **MUST NOT** silently fall back to `ei`. It raises `AcquisitionEvaluationError`.

---

## 3. Strict Candidate Identity & Finite Pool Semantics

In physical materials science and automated synthesis, candidates belong to finite physical pools (e.g. pre-synthesized library plates, discrete composition grids, batch formulations).

1. **Identity Preservation**:
   - Every candidate is identified by a unique string `candidate_id`.
   - Filtering of previously observed candidates occurs by `candidate_id`.
   - When `strict_identity=True` (default in `FiniteCandidatePool` and `ScientificClosedLoopCoordinator`), duplicate physical coordinates with distinct IDs are never erroneously filtered.
2. **Discrete Evaluation**:
   - Acquisition functions are evaluated directly over unobserved candidates in the finite pool:
     $$x^* = \arg\max_{x \in \mathcal{X}_{\text{unobserved}}} \alpha(x \mid \mathcal{D})$$
   - Continuous relaxation with nearest-neighbor projection is prohibited to prevent duplicate collisions and out-of-domain drift.
3. **Batch Semantics Transparency**:
   - Multi-candidate proposals ($n > 1$) generated by discrete top-$n$ scoring are explicitly tagged in `proposal_metadata`:
     ```json
     {
       "batch_semantics": "top_n_individual_scores",
       "batch_requested": 4
     }
     ```
   - They are never misrepresented as joint $q$-batch optimization.

---

## 4. Ledger Authority & Stateless Optimizer Model Lifecycle

1. **The Ledger is Authoritative**:
   - The SQLite `ExperimentLedger` is the single source of truth for all experimental facts, characterizations, and performance observations.
   - BoTorch surrogate model objects (`SingleTaskGP`, PyTorch tensors, optimizer states) are **stateless and ephemeral**. They are never serialized or persisted to the ledger database.
2. **Crash Recovery & Resume**:
   - Resuming from a ledger (`ScientificClosedLoopCoordinator.resume_from_ledger`) rebuilds state purely by re-reading completed records from the ledger.
   - BoTorch models are initialized and fitted on-demand from the reconstructed training frame at the exact moment a new proposal is requested.
3. **Snapshot Payloads**:
   - Optimizer snapshots saved to the ledger contain only backend-neutral JSON-serializable dictionaries (`backend_name`, `backend_version`, `strategy`, `proposal_sequence`, `step`, `current_best`, `dataset_fingerprint`).

---

## 5. Invariance & Reliability Guarantees

1. **Affine Scale Invariance**:
   - For any positive affine scaling of the target property ($y \to a \cdot y + b, a > 0$), candidate ranking under `greedy`, `gp_ucb`, `ei`, and `nei` is mathematically preserved.
   - Ensured via `Standardize(m=1)` in `BoTorchBackend`.
2. **Fail-Closed Objective Constraints**:
   - Any optimization objective specifying non-empty constraints raises `NotImplementedError` until explicit constrained acquisition math is implemented.

---

## 6. Maintenance Policy

1. **No Bespoke BO Implementations in `src/`**:
   - All legacy native BO code is quarantined under `src/legacy/native_optimizer/` for historical benchmark reference only.
   - No new custom GP kernels, custom acquisition functions, or heuristic controllers may be added to `src/optimization/`.
2. **External Library Upgrades**:
   - Future enhancements (e.g. multi-objective qEHVI, contextual BO, or Ax integration) must use official BoTorch/Ax library primitives wrapped via `OptimizerBackend`.

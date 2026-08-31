# Optimizer Infrastructure Freeze & Specification Reference

**Status**: FROZEN & ENFORCED  
**Version**: 1.0.0  
**Effective Date**: August 2026  
**Primary Backend**: BoTorch (`src.optimization.botorch_backend.BoTorchBackend`)  
**Freeze Target**: AIcoScientist Optimizer Core & Benchmark Infrastructure  

---

## 1. Executive Architecture Mandate

AIcoScientist is a scientific reasoning, materials/process characterization, and closed-loop experiment lifecycle platform. **AIcoScientist does not build, reinvent, or maintain generic Bayesian optimization infrastructure.**

All generic Bayesian optimization surrogate fitting, transforms, and acquisition calculations are strictly delegated to **BoTorch** (PyTorch / GPyTorch ecosystem).

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
|  - Canonical Strategy Resolution & Aliasing (resolve_strategy)                    |
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

The backend supports exactly 6 canonical strategies. Any unmapped or retired strategy is rejected immediately with `UnsupportedStrategyError`.

| Canonical Strategy Name | Accepted Aliases | Mathematical Implementation | BoTorch Primitive |
| :--- | :--- | :--- | :--- |
| `random` | `uniform`, `uniform_random` | Uniform random selection over unobserved candidates | Pseudo-random discrete sample |
| `greedy` | `posterior_mean` | Posterior mean prediction $\mu(x)$ | `model.posterior(X).mean` |
| `gp_ucb` | `ucb`, `upper_confidence_bound` | Upper Confidence Bound: $\mu(x) + \sqrt{\beta} \cdot \sigma(x)$ | Analytic UCB / Posterior variance |
| `expected_improvement` | `ei`, `log_ei`, `log_expected_improvement` | Analytic Expected Improvement over incumbent | `LogExpectedImprovement` / `ExpectedImprovement` |
| `noisy_expected_improvement` | `nei`, `q_nei`, `log_nei`, `log_noisy_expected_improvement` | Monte Carlo / Log Noisy Expected Improvement | `qLogNoisyExpectedImprovement` / `qNoisyExpectedImprovement` |
| `thompson` | `thompson_sampling`, `ts` | Joint discrete multivariate Gaussian posterior sample | `model.posterior(X).rsample()` |

### Prohibited & Retired Heuristics
- **No Fake TuRBO Labels**: `turbo_nei`, `turbo_ei`, `turbo`, and `adaptive` are retired and rejected with `UnsupportedStrategyError`.
- **No Silent Fallback**: If `noisy_expected_improvement` fails during evaluation, it **MUST NOT** silently fall back to `ei`. It raises `AcquisitionEvaluationError`.

---

## 3. Strict Candidate Identity & Finite Pool Semantics

In physical materials science and automated synthesis, candidates belong to finite physical pools (e.g. pre-synthesized library plates, discrete composition grids, batch formulations).

1. **Identity Preservation & Verification**:
   - Every candidate is identified by a unique string `candidate_id`.
   - Filtering of previously observed candidates occurs strictly by `candidate_id`.
   - When `strict_identity=True` (default in `FiniteCandidatePool` and `ScientificClosedLoopCoordinator`):
     - Every observation (whether `pd.DataFrame`, `Sequence[Mapping]`, `set[str]`, or `Sequence[str]`) MUST contain a non-null, valid physical `candidate_id`. Missing or null IDs immediately raise `ValueError`.
     - Duplicate physical coordinates with distinct IDs are never erroneously filtered.
2. **Canonical Candidate Pool Content Fingerprint**:
   - Deterministic SHA256 content fingerprinting over `candidate_id` + design variables (`compute_candidate_pool_fingerprint`).
   - Guarantees:
     - Invariant to row permutation (shuffling candidate pool rows preserves fingerprint).
     - Sensitive to any candidate ID mutation.
     - Sensitive to any numerical feature coordinate mutation.
3. **Discrete Evaluation**:
   - Acquisition functions are evaluated directly over unobserved candidates in the finite pool:
     $$x^* = \arg\max_{x \in \mathcal{X}_{\text{unobserved}}} \alpha(x \mid \mathcal{D})$$
   - Continuous relaxation with nearest-neighbor projection is prohibited.
4. **Batch Semantics Transparency**:
   - Multi-candidate proposals ($n > 1$) generated by discrete top-$n$ scoring are explicitly tagged in `proposal_metadata`:
     ```json
     {
       "batch_semantics": "top_n_individual_scores",
       "batch_requested": 4
     }
     ```
   - They are never misrepresented as joint $q$-batch optimization.

---

## 4. Ledger Authority & Snapshot Restoration / State Validation Contract

1. **The Ledger is Authoritative**:
   - The SQLite `ExperimentLedger` is the single source of truth for all experimental facts, characterizations, and performance observations.
   - BoTorch surrogate model objects (`SingleTaskGP`, PyTorch tensors, optimizer states) are **stateless and ephemeral**. They are never serialized or persisted to the ledger database.
2. **Crash Recovery & Resume (`resume_from_ledger`)**:
   - Resuming from a ledger rebuilds state purely by re-reading completed records from the ledger.
   - BoTorch models are initialized and fitted on-demand from the reconstructed training frame at the moment a proposal is requested.
3. **Snapshot Restoration & Validation Contract**:
   - Optimizer snapshots saved to the ledger contain:
     - `backend_name`, `backend_version`
     - `strategy`, `random_state`
     - `candidate_pool_fingerprint`, `dataset_fingerprint`
     - `target_col`, `feature_cols`, `objective`
   - **Contract on Resume**:
     - If caller omits `strategy`, `random_state`, or `backend`: the snapshot values are authoritatively restored.
     - If caller provides explicit conflicting parameters (`strategy`, `random_state`, `backend_name`, `target_col`, `feature_cols`), `ResumeStateMismatchError` is raised immediately.
     - If the candidate pool content fingerprint differs from the snapshot fingerprint, `ResumeStateMismatchError` is raised.
     - If runtime backend version differs from snapshot, a clear warning is logged and runtime version recorded.

---

## 5. Invariance & Reliability Guarantees

1. **Affine Scale Invariance**:
   - For any positive affine scaling of the target property ($y \to a \cdot y + b, a > 0$), candidate ranking and top proposal selection under `greedy`, `gp_ucb`, `expected_improvement`, `noisy_expected_improvement`, and `thompson` is mathematically preserved under fixed stochastic seed.
   - Ensured via `Standardize(m=1)` in `BoTorchBackend`.
2. **Fail-Closed Objective Semantics**:
   - Any optimization objective specifying non-empty constraints, multiobjective targets, or threshold semantics raises `NotImplementedError`.

---

## 6. Historical Reproduction Modules & Boundary Enforcement

1. **Active Production Path**:
   - Production coordinator (`ScientificClosedLoopCoordinator`), production benchmarks (`auirh_benchmark.py`, `feconi_benchmark.py`, `attia_continuous_benchmark.py`), and CLI demos have **zero imports** from `src.legacy`.
2. **Historical Baseline Reproduction**:
   - `src/evaluation/auirh_reproduction_benchmark.py` and `src/evaluation/feconi_reproduction_benchmark.py` are explicitly marked as historical reference modules. They import `src.legacy.native_optimizer` strictly to reproduce superseded legacy results and preserve scientific auditability.

---

## 7. Dependency Matrix & Frozen Environment Lock

| Package | Tested Compatibility Range (`requirements-core.txt`) | Frozen Benchmark Lock (`requirements-benchmark-lock.txt`) |
| :--- | :--- | :--- |
| `botorch` | `>=0.11.0,<1.0.0` | `0.18.1` |
| `torch` | `>=2.2.0,<3.0.0` | `2.13.0+cpu` |
| `gpytorch` | `>=1.11.0,<2.0.0` | `1.15.2` |
| `pandas` | `>=2.0.0,<4.0.0` | `3.0.3` |
| `numpy` | `>=1.24.0,<3.0.0` | `2.4.6` |
| `scipy` | `>=1.10.0,<2.0.0` | `1.18.0` |
| `scikit-learn` | `>=1.3.0,<2.0.0` | `1.9.0` |

---

## 8. Maintenance Policy & Freeze Sign-off

1. **No Bespoke BO Implementations in `src/`**:
   - All legacy native BO code is quarantined under `src/legacy/native_optimizer/`.
   - No custom GP kernels, custom acquisition math, or heuristic controllers may be introduced.
2. **Future Enhancements**:
   - Any future optimization capabilities (e.g. multi-objective qEHVI, contextual BO, or Ax integration) must use official BoTorch/Ax library primitives wrapped via `OptimizerBackend`.

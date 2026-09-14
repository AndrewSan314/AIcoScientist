# Battery Process Stress Suite

Status: **PARTIAL**.

## Architecture status

- scalar_contextual_stage_optimizer: **IMPLEMENTED_NOT_VALIDATED** — Context-conditioned stage-wise optimizer is implemented and unit-tested; source-backed multi-stage trajectory validation remains unavailable.
- multimodal_contextual_state: **IMPLEMENTED_NOT_VALIDATED** — Multimodal latent state adapter is implemented and unit-tested; no source-backed trained state benchmark is claimed.
- horizon_safe_evidence_replay: **IMPLEMENTED_NOT_VALIDATED** — Horizon-safe source-backed evidence replay is implemented and unit-tested; no adaptive policy performance is claimed.
- adaptive_evidence_policy: **IMPLEMENTED_NOT_VALIDATED** — Cost-aware heuristic selection and blinded source-backed reveal are implemented and unit-tested; no source-backed policy-performance or EVI claim is made.

## Source-backed results

- Warwick ultrasound grouped holdout (48 rows/24 groups): gated R²=0.792; naive concatenation R²=0.943. Gated fusion is not superior on this split. Derived ultrasound-masking stress rates: 10%, 25%, 50%, 75%.

## Offline replay

- drakopoulos_graphite expected_improvement: 3 seeds, mean final simple regret=0.
- drakopoulos_graphite gp_ucb: 3 seeds, mean final simple regret=0.
- drakopoulos_graphite greedy: 3 seeds, mean final simple regret=1.17487.
- drakopoulos_graphite noisy_expected_improvement: 3 seeds, mean final simple regret=0.
- drakopoulos_graphite random: 3 seeds, mean final simple regret=1.00225.
- drakopoulos_graphite thompson: 3 seeds, mean final simple regret=1.33963.
- warwick_nmc622 expected_improvement: 3 seeds, mean final simple regret=0.
- warwick_nmc622 gp_ucb: 3 seeds, mean final simple regret=0.
- warwick_nmc622 greedy: 3 seeds, mean final simple regret=0.
- warwick_nmc622 noisy_expected_improvement: 3 seeds, mean final simple regret=0.
- warwick_nmc622 random: 3 seeds, mean final simple regret=0.0833288.
- warwick_nmc622 thompson: 3 seeds, mean final simple regret=0.0905016.

## Reproducibility

- Command: `scripts/run_battery_process_benchmark.py --output C:\Users\ADMIN\AppData\Local\Temp\aicoscientist-structured-stress-20260914 --allow-unavailable --replay-seeds 3 --replay-steps 2`
- Commit: `53c852e7cd77f41da59f45649096131e6a0026d6`
- Python: `3.14.2`
- Process tests (run separately): `python -m pytest -q tests/process -p no:cacheprovider`
- Artifact manifest: `manifest.json`

## Limitations

- Small, source-specific datasets; no chemistry-generalization claim.
- Missing modalities are omitted rather than imputed.
- Unavailable raw sources: artistic.
- No live production control or causal-effect claim.

# ARTISTIC multi-fidelity claim boundaries

ARTISTIC has two explicit execution modes:

- `REFERENCE`: the pinned upstream slurry horizon, exactly 20,000,000 steps.
- `SHORT_HORIZON`: an explicit positive horizon below 20,000,000, labeled `SIMULATED_STRESS` and lower fidelity.

The pinned checkout is immutable. A short run is rendered into an isolated workspace and records the exact input replacement, original/new text, and before/after hashes. The upstream tree is never edited. Fidelity mode, requested/completed steps, checkpoint interval, progress, early termination, and a fidelity identity are written to the run manifest and carried into the normalized cache entry.

Short runs are useful for smoke tests, stress behavior, progress diagnostics, and convergence planning. They are not substitutes for the reference horizon and cannot claim equivalence to it. A convergence report requires both at least three internally stable metric-bearing checkpoints and a checkpoint at exactly 20M steps from the reference run. It records internal stability separately from reference agreement, and validates only when every required metric has an explicit typed tolerance policy: relative per metric or default, with an absolute tolerance for near-zero reference values. Missing metrics, undefined tolerances, far references, and short references remain nonvalidated.

Thermo progress is parsed only from `slurry.log`; checkpoint records retain finite metric values, stage, source log, and malformed-row diagnostics. Reference execution is opt-in in the programmatic API (`confirm_reference_execution=True`); short-horizon execution is unaffected. The study planner remains dry-run planning only.

The study planner exposes 500k, 1M, 2M, 5M, 10M, and 20M horizons with optional measured runtime estimates. It is dry-run planning only; it does not launch an automatic sweep. Exact reference execution requires explicit confirmation in the CLI.

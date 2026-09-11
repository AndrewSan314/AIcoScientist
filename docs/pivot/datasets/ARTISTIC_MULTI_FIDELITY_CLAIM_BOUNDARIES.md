# ARTISTIC multi-fidelity claim boundaries

ARTISTIC has two explicit execution modes:

- `REFERENCE`: the pinned upstream slurry horizon, exactly 20,000,000 dynamics-relative slurry steps.
- `SHORT_HORIZON`: an explicit positive horizon below 20,000,000, labeled `SIMULATED_STRESS` and lower fidelity.

The pinned checkout is immutable. A short run is rendered into an isolated workspace and records the exact input replacement, original/new text, and before/after hashes. The upstream tree is never edited. Fidelity mode, requested/completed steps, checkpoint interval, progress, early termination, and a fidelity identity are written to the run manifest and carried into the normalized cache entry.

Short runs are useful for smoke tests, stress behavior, progress diagnostics, and convergence planning. They are not substitutes for the reference horizon and cannot claim equivalence to it. A convergence report requires typed `ConvergenceRunEvidence` for both runs, identical recipe/source-tree/physics identities, at least three internally stable metric-bearing short checkpoints, and a successful reference checkpoint at exactly 20M dynamics-relative steps. It records internal stability separately from reference agreement, and validates only when every required metric has an explicit typed tolerance policy: relative per metric or default, with an absolute tolerance for near-zero reference values. Missing metadata, metrics, undefined tolerances, far references, and short references remain nonvalidated.

Thermo progress is parsed only from `slurry.log`; checkpoint records retain the raw LAMMPS `Step`, a dynamics-relative step, phase, finite metric values, stage, source log, and malformed-row diagnostics. Minimization rows are never treated as dynamics progress. If phase timing cannot be separated, ETA/rate fields are explicitly unavailable. The reference horizon is 20M dynamics steps, not necessarily raw LAMMPS step 20M. Reference execution is opt-in in the programmatic API (`confirm_reference_execution=True`); short-horizon execution is unaffected. The study planner remains dry-run planning only.

The default study planner prepares only 500k, 1M, and 2M dry-run entries with optional measured runtime estimates. Longer horizons, including 5M, 10M, and 20M, require an explicit command. It never launches an automatic sweep; exact reference execution requires explicit confirmation in the CLI.

# ARTISTIC multi-fidelity claim boundaries

ARTISTIC has two explicit execution modes:

- `REFERENCE`: the pinned upstream slurry horizon, exactly 20,000,000 steps.
- `SHORT_HORIZON`: an explicit positive horizon below 20,000,000, labeled `SIMULATED_STRESS` and lower fidelity.

The pinned checkout is immutable. A short run is rendered into an isolated workspace and records the exact input replacement, original/new text, and before/after hashes. The upstream tree is never edited. Fidelity mode, requested/completed steps, checkpoint interval, progress, early termination, and a fidelity identity are written to the run manifest and carried into the normalized cache entry.

Short runs are useful for smoke tests, stress behavior, progress diagnostics, and convergence planning. They are not substitutes for the reference horizon and cannot claim equivalence to it. A convergence report may compare only metrics present in the ARTISTIC outputs/logs. Without an exact 20M-step reference it reports `REFERENCE_NOT_AVAILABLE`; without enough checkpoints it reports `INSUFFICIENT_CHECKPOINTS`. Stability is diagnostic (`STABILITY_OBSERVED` or `NOT_STABLE`), never an automatic stop or a scientific equivalence claim.

The study planner exposes 500k, 1M, 2M, 5M, 10M, and 20M horizons with optional measured runtime estimates. It is dry-run planning only; it does not launch an automatic sweep. Exact reference execution requires explicit confirmation in the CLI.

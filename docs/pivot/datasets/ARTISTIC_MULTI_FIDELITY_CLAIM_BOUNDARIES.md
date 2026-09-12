# ARTISTIC multi-fidelity claim boundaries

ARTISTIC has two explicit execution modes:

- `REFERENCE`: the pinned upstream slurry horizon, exactly 20,000,000 dynamics-relative slurry steps.
- `SHORT_HORIZON`: an explicit positive horizon below 20,000,000, labeled `SIMULATED_STRESS` and lower fidelity.

The pinned checkout is immutable. A short run is rendered into an isolated workspace and records the exact input replacement, original/new text, and before/after hashes. The upstream tree is never edited. Fidelity mode, requested/completed steps, checkpoint interval, progress, early termination, and a fidelity identity are written to the run manifest and carried into the normalized cache entry.

Short runs are useful for smoke tests, stress behavior, progress diagnostics, and convergence planning. They are not substitutes for the reference horizon and cannot claim equivalence to it. A convergence report requires typed `ConvergenceRunEvidence` for both runs, identical recipe/source-tree/physics identities, a `StabilityPolicy`-satisfied set of metric-bearing short checkpoints, and a successful reference checkpoint at exactly 20M dynamics-relative steps. The default diagnostic policy requires 10 checkpoints; callers may choose a stricter or explicitly documented policy for a test. Stability is recorded separately from reference agreement, and validates only when every required metric has an explicit typed tolerance policy: relative per metric or default, with an absolute tolerance for near-zero reference values. These software thresholds are diagnostic safeguards, not scientifically validated tolerances. Missing metadata, metrics, undefined tolerances, far references, and short references remain nonvalidated.

Thermo progress is parsed only from `slurry.log`; checkpoint records retain the raw LAMMPS `Step`, a dynamics-relative step, phase, finite metric values, stage, source log, and malformed-row diagnostics. Minimization rows are never treated as dynamics progress. If phase timing cannot be separated, ETA/rate fields are explicitly unavailable. The reference horizon is 20M dynamics steps, not necessarily raw LAMMPS step 20M. Reference execution is opt-in in the programmatic API (`confirm_reference_execution=True`); short-horizon execution is unaffected. The study planner remains dry-run planning only.

The default study planner prepares only 500k, 1M, and 2M dry-run entries with optional measured runtime estimates. For the published recipe, the explicit no-launch preparation command is:

```powershell
python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --study-dry-run --mode mpi --mpi-processes 4 --steps-per-second 1000
```

The base config is intentionally left in its valid `REFERENCE` mode; the planner assigns `SHORT_HORIZON` to 500k, 1M, and 2M entries. It reports 21,517 predicted particles for each horizon, the requested dynamics steps, dump/thermo interval, pinned source identities, a shared `physics_config_fingerprint`, and proportional runtime estimates when a measured rate is supplied. It never launches an automatic sweep.

After reviewing each preflight result, run the horizons explicitly with stable run IDs:

```powershell
python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --run-id artistic-short-500k --fidelity-mode SHORT_HORIZON --slurry-steps 500000 --mode mpi --mpi-processes 4 --preflight
python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --run-id artistic-short-500k --fidelity-mode SHORT_HORIZON --slurry-steps 500000 --mode mpi --mpi-processes 4

python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --run-id artistic-short-1m --fidelity-mode SHORT_HORIZON --slurry-steps 1000000 --mode mpi --mpi-processes 4 --preflight
python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --run-id artistic-short-1m --fidelity-mode SHORT_HORIZON --slurry-steps 1000000 --mode mpi --mpi-processes 4

python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --run-id artistic-short-2m --fidelity-mode SHORT_HORIZON --slurry-steps 2000000 --mode mpi --mpi-processes 4 --preflight
python scripts/run_artistic_simulator.py config/artistic/published_reference_recipe.json --run-id artistic-short-2m --fidelity-mode SHORT_HORIZON --slurry-steps 2000000 --mode mpi --mpi-processes 4
```

These commands record the selected MPI mode and fidelity identities; they do not claim equivalence to the 20M reference. Exact reference execution still requires explicit confirmation in the CLI.

# Real ARTISTIC 500k slurry cutoff

## Scope and status

- Run: `artistic-short-500k-rerun`; SHORT_HORIZON slurry only.
- Slurry completion: 500,000/500,000 dynamics-relative steps; raw LAMMPS 287,300 → 787,300; 35 parsed thermo checkpoints and no parser diagnostics.
- The canonical full-pipeline manifest is `Numerical failure` because the downstream drying command was intentionally terminated. The cutoff sidecar classifies the completed stage as `SLURRY_CUTOFF_SUCCESS`.
- Drying/calendering are incomplete. The available drying dumps are audit-only, excluded from final-KPI and training claims.

## Observations

- Final slurry density: 1.4788396857021.
- Total slurry command wall time: 3.889 h using 4 MPI ranks. The effective total-stage rate is 35.7168 completed dynamics steps per total slurry-command wall second; pure dynamics timing is unavailable.
- Rough total-stage linear projection: 20,000,000 steps is 6.48 days from start (6.32 additional days after this run). This projection linearly scales the observed 500k total slurry-command walltime and may overestimate longer horizons because minimization is a fixed cost.
- Existing diagnostic stability policy: last 10 checkpoints, 1% relative tolerance, all available metrics. Result: `NOT_STABLE`. The official policy is preserved; `tail_stability_metrics.csv` adds safe zero-crossing/near-zero and recent-window diagnostics only.
- Reference equivalence: `REFERENCE_NOT_AVAILABLE`; no compatible exact 20,000,000-step reference was supplied.

## Provenance

- Recipe fingerprint: `a768b6d5ebbdab98d1ab92dc262e59990f64e30f663a5dbb67a22c19e98a22f9`
- Physics fingerprint: `265febb10fbcba756a0484fa91eb75f7d69267f5a4e6d81e0315088edea69a0e`
- Pinned source commit/tree: `5af9e0345673fac557c479ee8f4a0727e442c1fa` / `e870c018acf8696424ae3ac9c36163f5bd3afb56`
- MPI environment was validated: `True`.
- SHA-256 bindings for the manifest, cutoff sidecar, rendered slurry input, log, density, and final slurry coordinates are in `summary.json`.

Artifacts: `thermo_checkpoints.csv`, `tail_stability_metrics.csv`, `runtime_projection.json`, and `summary.json`.

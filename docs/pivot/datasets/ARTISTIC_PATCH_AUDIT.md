# ARTISTIC patch audit

The immutable upstream checkout is never edited. Compatibility patches are applied only to the copied per-run workspace and are recorded with before/after SHA-256 values in its manifest.

| Finding | Decision | Evidence |
| --- | --- | --- |
| `Slurry/init_structure.txt` computes `n_AM7..10` with `p_AM6` | Apply the verified workspace-only `slurry_am_fraction_indices` patch | lines 85–88 use `p_AM6`; each count must use its own declared `p_AM7..10` fraction |
| `Drying_heterogeneous/in_evaporation_freeze.run` computes `volume_CBD3` with `n_CBD2` | Do not patch yet | the following `volume_CBD` definition is overwritten at line 147 before downstream use; no output effect is established without a real regression run |
| `Calendering/user_inputs_cal.txt` says `minimize=1` means no minimization, but `in_cal.run` minimizes when it equals 1 | Expose `perform_energy_minimization` and render its actual executable meaning | runner preserves the `in_cal.run` conditional; the mismatch is captured in each manifest |

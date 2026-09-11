# ARTISTIC source-to-schema mapping

The public pinned source is rendered only at its documented `@input@` placeholders. No extra optimization controls are invented.

`SlurryRecipe.electrode_mass_ug` is schema v3's explicit unit name. It renders unchanged as upstream `@dry_mass@`; `Slurry/user_inputs.txt` performs the exact conversion `@dry_mass@/1E6` to grams. The old `dry_mass_mg` name is rejected by the runner rather than reinterpreted. Existing normalized records retain their v2 adapter/schema provenance.

| Expected ARTISTIC source evidence | Intended normalized destination |
| --- | --- |
| `Slurry/user_inputs.txt`: AM population, diameters/fractions, CBD diameter, solid content, mass, AM/CBD ratios, thickness, nanoporosity | `MIXING.controls` |
| `Slurry/density_slurry.out`, `coord_out_slurry.data` | `MIXING.intermediate_properties`; lineage artifact |
| `Drying_heterogeneous/user_inputs_evHet.txt`: zone count, evaporation mode/rates | `DRYING.controls` |
| `AM_loading.out`, `coord_out_electrode.data`, `porosity_bulk.out`, `porosity_all.out` | `DRYING.intermediate_properties`; lineage artifact |
| `Calendering/user_inputs_cal.txt`: compression, CBD nanoporosity decrease, relaxation, minimization | `CALENDERING.controls` |
| `initial_lz` | diagnostic only; never a training feature or target |
| `Cal_electrode.atom` particle positions and its dump `zlo`; `new_CBD_nanoporosity`, `porosity_cal_bulk.out`, `porosity_cal_all.out` | final simulated KPIs only; never `CALENDERING.intermediate_properties` |

`calendered_electrode_thickness` is `max(final particle z) - dump zlo`, matching the physical extent used by upstream `pores_cal.py` (`zmax` with the lower z boundary). `coord_out_cal.data` keeps a fixed simulation container during wall compression, so its z bounds are recorded only as `simulation_box_z_length` diagnostic and are never used as electrode thickness.

The same physical value cannot occupy both an intermediate-property and final-KPI role. `check.txt` and `check_cal.txt` are required non-empty post-processing artifacts, not invented numeric observables.

Raw source hashes, rendered-input hashes, patch records, executable versions, commands, output hashes, and boundary-to-boundary lineage hashes are written into each isolated `outputs/artistic_runs/<run_id>/manifest.json`.

Schema v3 accepts the published Online Calculator domain (Lombardo et al., *Batteries & Supercaps*, 2022, DOI 10.1002/batt.202100324): electrode mass 0.1–0.2 ug; CBD diameter 0.7–1.5 um; CBD nanoporosity 0.3–0.7; active AM diameters 2–25 um; solid content 0.42–0.70; AM/CBD 0.85–0.97/0.03–0.15; and calendering compression 0.05–0.40. The source exposes aggregate CBD, so its carbon/binder sub-split is not invented.

Before rendering or execution, `ArtisticSimulator.preflight()` derives AM/CBD particle counts with the source mass/density arithmetic (`pi = 3.1415926` and its observed integer evaluation) and applies the configurable `max_particle_count` guard (default 1,000,000). `scripts/run_artistic_simulator.py <recipe> --preflight` reports the count and conservative memory class without invoking LAMMPS.

Use `--mode mpi --mpi-processes N` for an MPI execution; Windows defaults to `mpiexec`, while POSIX defaults to `mpirun`.

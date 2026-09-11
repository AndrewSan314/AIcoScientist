# ARTISTIC source-to-schema mapping

The public pinned source is rendered only at its documented `@input@` placeholders. No extra optimization controls are invented.

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

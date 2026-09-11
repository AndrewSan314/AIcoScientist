# ARTISTIC source-to-schema mapping

The public pinned source is rendered only at its documented `@input@` placeholders. No extra optimization controls are invented.

| Expected ARTISTIC source evidence | Intended normalized destination |
| --- | --- |
| `Slurry/user_inputs.txt`: AM population, diameters/fractions, CBD diameter, solid content, mass, AM/CBD ratios, thickness, nanoporosity | `MIXING.controls` |
| `Slurry/density_slurry.out`, `coord_out_slurry.data` | `MIXING.intermediate_properties` |
| `Drying_heterogeneous/user_inputs_evHet.txt`: zone count, evaporation mode/rates | `DRYING.controls` |
| `AM_loading.out`, `coord_out_electrode.data` | `DRYING.intermediate_properties` |
| `Calendering/user_inputs_cal.txt`: compression, CBD nanoporosity decrease, relaxation, minimization | `CALENDERING.controls` |
| `initial_lz`, `new_CBD_nanoporosity`, `coord_out_cal.data` | `CALENDERING.intermediate_properties` and final simulated KPIs |

Raw source hashes, rendered-input hashes, patch records, executable versions, commands and parsed output hashes are written into each isolated `outputs/artistic_runs/<run_id>/manifest.json`.

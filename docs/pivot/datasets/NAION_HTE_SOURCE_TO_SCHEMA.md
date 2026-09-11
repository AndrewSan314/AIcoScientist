# Na-ion high-throughput source-to-schema mapping

| Source evidence | Normalized destination |
| --- | --- |
| CSV filename | stable `run_id`; filename cell token becomes `cell_id` |
| Top-level archive directory (for example `C20Form`) | `batch_id` and source protocol context |
| First 256 valid rows: cycle/time/current/voltage/capacity columns | `FORMATION_CURVE` at `FORMATION` |
| Last 256 valid rows from the same CSV | `CYCLING_CURVE` at `FINAL_CHARACTERIZATION` |
| Last non-empty `Discharge_Capacity(Ah)` | `last_observed_discharge_capacity_ah` |

All source paths and the archive SHA-256 are attached in provenance. The adapter intentionally does not populate formulation, coating, or assembly controls because no audited source table links them to the individual cycling files.

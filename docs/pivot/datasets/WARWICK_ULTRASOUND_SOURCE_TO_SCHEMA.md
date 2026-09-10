# Warwick ultrasound V4 source-to-schema mapping

| Source field | Normalized destination |
| --- | --- |
| `metadata.Sample_ID` and material directory | `BatteryProcessRun.run_id`, `chemistry_id` |
| Numeric calendering metadata except state, thickness, density | `CALENDERING.controls` |
| Before `metadata.Thickness` | `pre_calendering_thickness_um` |
| After `metadata.Thickness` | `post_calendering_thickness_um` |
| After `metadata.Density` | `post_calendering_density_g_cm3` when supplied |
| `fft_frequency`, `fft_magnitude` | paired `ULTRASOUND_SPECTRUM` modality observations |

The source paths and archive SHA are recorded in each run provenance. Exact normalized control tuples define grouping splits.

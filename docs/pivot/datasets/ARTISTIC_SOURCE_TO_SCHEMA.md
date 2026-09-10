# ARTISTIC source-to-schema status

No source-to-schema field mapping exists yet because the record files are restricted. The reserved mapping contract is:

| Expected ARTISTIC source evidence | Intended normalized destination |
| --- | --- |
| `user_inputs*` slurry settings | `MIXING.controls` |
| Drying input/output fields | `DRYING.controls` and intermediate properties |
| Calendering input/output fields | `CALENDERING.controls` and final simulated KPIs |
| Source-linked 3D electrode output | `XCT_VOLUME` modality with `SIMULATED_PHYSICS` provenance |

Populate this table only after authorized raw files are downloaded, hashed, and audited.

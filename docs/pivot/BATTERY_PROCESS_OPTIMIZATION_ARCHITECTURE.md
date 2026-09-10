# Battery process optimization architecture

`BatteryProcessRun` is an ordered list of typed `StageRecord`s. A stage separates controllable settings from observations and represents absence explicitly with a missing modality record. `InformationHorizon` materializes only pre-decision records; final KPIs and future stages are forbidden.

```text
source-backed raw data -> adapter -> BatteryProcessRun -> InformationHorizon
  -> tabular/signal/image/curve encoders -> masked gated fusion
  -> stage transition + multi-output predictions + uncertainty
  -> feasible process search space -> official BoTorch finite-pool proposal
  -> offline reveal/replay -> immutable result manifest
```

The first implementation is electrode scoped: formulation, mixing, coating, drying, calendering, and characterization. The stage enum still covers the complete cell-manufacturing chain. A generic finite-pool BO backend remains the production acquisition primitive; process contracts compose it rather than duplicating generic BO mathematics.

Evidence labels are mandatory: `PHYSICAL_HISTORICAL`, `PILOT_LINE_HISTORICAL`, `SIMULATED_PHYSICS`, `SIMULATED_STRESS`, `OFFLINE_REPLAY`, and `LIVE_PROCESS`. Simulated results are never physical validation.

# Battery process optimization architecture

`BatteryProcessRun` is an ordered list of typed `StageRecord`s. A stage separates controllable settings from observations and represents absence explicitly with a missing modality record. Scalar training views pair every absent numeric field's zero placeholder with a `__observed` mask, so the placeholder is never treated as a measured physical zero. `InformationHorizon` materializes only pre-decision records; final KPIs and future stages are forbidden.

```text
source-backed raw data -> adapter -> BatteryProcessRun -> InformationHorizon
  -> existing tabular/signal/image/curve encoders -> GatedMaskedFusion
  -> MASPOProcessStateModel -> StageAwareProcessModel
  -> stage-scoped predictions + uncertainty
  -> feasible process search space -> official BoTorch finite-pool proposal
  -> MASPO receding-horizon next-stage action -> blinded evidence replay
  -> robustness report -> immutable result manifest
```

The first implementation is electrode scoped: formulation, mixing, coating, drying, calendering, and characterization. The stage enum still covers the complete cell-manufacturing chain. A generic finite-pool BO backend remains the production acquisition primitive; process contracts compose it rather than duplicating generic BO mathematics.

`MASPOProcessOptimizationCoordinator` commits only the next legal downstream
stage. It retains the remaining horizon for later decisions, measures decision
latency, and accepts either a validated `BatteryProcessRun` or an already
projected `HorizonView`. `MASPOProcessStateModel` batches masked modalities,
then exposes final and stage-scoped intermediate heads without exposing future
records to a pre-decision model.

The evidence replay path selects an acquisition option before revealing its
source-backed observation. Cost and latency are recorded; unavailable or
synthetic evidence is rejected. Robustness evaluation consumes caller-supplied
bounded perturbations and reports worst case, variance, feasibility, and
constraint-violation rates. It does not invent perturbations or alter the
official acquisition backend.

ARTISTIC is a `SIMULATED_PHYSICS` stress dataset. Its LAMMPS/MPI execution,
particle-count convention, streamed logs, and normalized-cache manifest must
be validated before it can enter training. A successful ARTISTIC run does not
become physical evidence.

Evidence labels are mandatory: `PHYSICAL_HISTORICAL`, `PILOT_LINE_HISTORICAL`, `SIMULATED_PHYSICS`, `SIMULATED_STRESS`, `OFFLINE_REPLAY`, and `LIVE_PROCESS`. Simulated results are never physical validation.

# A-Lab Multimodal Validation

- Available ledger samples: 1035
- Clean controlled worlds: `METHODOLOGY_VALID`
- Stress controlled worlds: `METHODOLOGY_VALID`
- Full policy matrix: `METHODOLOGY_VALID` (180 trajectories)
- Retrospective replay: `METHODOLOGY_VALID`
- Scientific methodology: `CONTROLLED_CLEAN_AND_STRESS_METHODOLOGY_VALIDATED`
- Release readiness: `PENDING_EXTERNAL_CI` (external CI: `NOT_INSPECTED`)
- Clean distribution consistency: `PASS`
- Shared structural nuisance: `PASS`
- Posterior concentration warning: `PASS` (sample-product evidence is chemistry-correlated)
- Sample interpolation: `SAMPLE_ID_INTERPOLATION_HOLDOUT` (510 calibration / 525 evaluation)
- Reaction group holdout: `REACTION_SIGNATURE_GROUP_HOLDOUT`
- Target holdout: `TARGET_COMPOUND_GROUP_HOLDOUT`
- SEM/EDS candidate actions: disabled because archives are precursor-level and not canonically linked to sample IDs.
- Scope: retrospective historical replay only; no prospective or causal claim.
- H1 structural metrics are held-out evaluation metrics; H2/H3 mechanistic components remain explicitly weakly identified or not identifiable.
- Posterior values are relative explanatory model weights among simplified competing models, not probabilities that exactly one mechanism is true.

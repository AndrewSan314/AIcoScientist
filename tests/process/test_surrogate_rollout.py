from __future__ import annotations

import numpy as np
import pytest

from scripts.run_maspo_pipeline import synthetic_frame
from src.process.surrogates import GenericTabularAdapter, ProcessSurrogate, SurrogateArtifact, SurrogateDecisionContext, SurrogateInputSchema, one_stage_lookahead, split_groups
from src.process.surrogates.core import TrainOnlyPreprocessor


MAPPING = {"dataset_name": "rollout-fixture", "dataset_version": "1", "sample_id": "sample_id", "run_id": "run_id", "recipe_id": "recipe_id", "group_id": "group_id", "stage": "stage", "fidelity": "fidelity", "requested_horizon": "requested_horizon", "controls": ["cbd_fraction", "calender_pressure"], "observations": ["drying_porosity"], "previous_state": ["slurry_viscosity"], "modality_state": [], "targets": ["capacity_retention"]}


def _artifact() -> tuple[SurrogateArtifact, SurrogateDecisionContext]:
    adapter = GenericTabularAdapter(synthetic_frame(), MAPPING, source="synthetic-test")
    samples, manifest = adapter.samples(), adapter.manifest()
    split = split_groups(samples, manifest, seed=42)
    index = {sample.sample_id: sample for sample in samples}
    train = [index[item] for item in split.train_ids]
    schema = SurrogateInputSchema.from_training_samples(train, declared_fidelities=manifest.fidelity_levels)
    preprocessor = TrainOnlyPreprocessor().fit(train, schema)
    surrogate = ProcessSurrogate("extra_trees", seed=42).fit(preprocessor.transform(train), {"capacity_retention": np.array([sample.targets["capacity_retention"] for sample in train])})
    artifact = SurrogateArtifact(surrogate, preprocessor, manifest.dataset_fingerprint, split.split_fingerprint, ("capacity_retention",), schema)
    sample = train[0]
    return artifact, SurrogateDecisionContext(sample.stage, sample.previous_state, sample.observations, sample.modality_state, sample.fidelity, sample.requested_horizon)


def test_stage_specific_rollout_is_explicitly_one_stage_and_fingerprinted() -> None:
    artifact, context = _artifact()
    rollout = one_stage_lookahead(artifact, context, [{"cbd_fraction": 0.03, "calender_pressure": 40.0}])
    assert rollout.mode == "ONE_STAGE_LOOKAHEAD"
    assert rollout.context_fingerprint == context.context_fingerprint
    assert len(rollout.predictions["capacity_retention"]["mean"]) == 1
    with pytest.raises(ValueError, match="Monte Carlo"):
        one_stage_lookahead(artifact, context, [{"cbd_fraction": 0.03, "calender_pressure": 40.0}], monte_carlo_samples=10)

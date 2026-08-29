from __future__ import annotations

import pytest

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.datasets.dynamic_cycling import DynamicCyclingAdapter
from src.datasets.severson import SeversonAdapter
from src.datasets.si_mxene_spec import SI_MXENE_SPEC


def test_scientific_data_contract_fields_si_mxene() -> None:
    spec = SI_MXENE_SPEC
    assert len(spec.pre_experiment_features) > 0
    assert len(spec.post_experiment_characterization) > 0
    assert len(spec.targets) > 0
    assert len(spec.candidate_variables) > 0

    # Ensure pre-experiment features and post-experiment characterization are mutually exclusive
    overlap = set(spec.pre_experiment_features) & set(spec.post_experiment_characterization)
    assert len(overlap) == 0

    # Candidate variables must only come from pre-experiment features
    cand_post_overlap = set(spec.candidate_variables) & set(spec.post_experiment_characterization)
    assert len(cand_post_overlap) == 0


def test_staged_feature_visibility_and_learning() -> None:
    spec = SI_MXENE_SPEC

    vis_pre = spec.optimizer_visible_features(stage="pre_experiment")
    assert len(vis_pre) > 0
    for feat in vis_pre:
        assert feat not in spec.post_experiment_characterization

    learn_post = spec.learning_features(stage="post_observation")
    assert set(learn_post) == set(spec.feature_columns)

    with pytest.raises(ValueError, match="Unknown workflow stage"):
        spec.optimizer_visible_features(stage="invalid_stage")


def test_two_stage_model_spec_contract() -> None:
    two_stage = TwoStageModelSpec(
        dataset_name="si_mxene",
        process_features=["sintering_temp", "etching_time"],
        characterization_targets=["sem_porosity", "particle_size"],
        performance_targets=["retention_100"],
    )
    assert two_stage.dataset_name == "si_mxene"
    assert len(two_stage.process_features) == 2
    assert len(two_stage.characterization_targets) == 2
    assert len(two_stage.performance_targets) == 1


def test_dataset_spec_leakage_guards_prevent_post_experiment_in_candidates() -> None:
    # 1. Candidate variables containing post-experiment feature
    with pytest.raises(ValueError, match="candidate_variables and post_experiment_characterization must not overlap"):
        DatasetSpec(
            name="leakage_spec",
            id_column="id",
            feature_columns=["temp", "sem_porosity"],
            target_column="capacity",
            candidate_columns=["temp"],
            pre_experiment_features=["temp"],
            post_experiment_characterization=["sem_porosity"],
            candidate_variables=["temp", "sem_porosity"],
            targets=["capacity"],
        )

    # 2. Candidate columns containing post-experiment feature
    with pytest.raises(ValueError, match="candidate_columns and post_experiment_characterization must not overlap"):
        DatasetSpec(
            name="leakage_spec",
            id_column="id",
            feature_columns=["temp", "sem_porosity"],
            target_column="capacity",
            candidate_columns=["temp", "sem_porosity"],
            pre_experiment_features=["temp"],
            post_experiment_characterization=["sem_porosity"],
            candidate_variables=["temp"],
            targets=["capacity"],
        )


def test_dataset_spec_leakage_guards_prevent_pre_post_overlap() -> None:
    with pytest.raises(ValueError, match="pre_experiment_features and post_experiment_characterization must not overlap"):
        DatasetSpec(
            name="leakage_spec",
            id_column="id",
            feature_columns=["temp", "sem_porosity"],
            target_column="capacity",
            candidate_columns=["temp"],
            pre_experiment_features=["temp", "sem_porosity"],
            post_experiment_characterization=["sem_porosity"],
            candidate_variables=["temp"],
            targets=["capacity"],
        )


def test_severson_and_dynamic_cycling_contract_fields() -> None:
    sev_adapter = SeversonAdapter()
    assert len(sev_adapter.spec.targets) == 1
    assert sev_adapter.spec.targets[0] == "cycle_life"
    assert len(sev_adapter.spec.post_experiment_characterization) > 0

    dc_adapter = DynamicCyclingAdapter()
    assert len(dc_adapter.spec.targets) == 1
    assert len(dc_adapter.spec.pre_experiment_features) > 0

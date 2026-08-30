from __future__ import annotations

import pytest

from src.datasets.base import DatasetSpec
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.validation import InformationHorizonError, validate_record_against_spec


@pytest.fixture
def sample_spec() -> DatasetSpec:
    return DatasetSpec(
        name="battery_lab_v1",
        id_column="sample_id",
        feature_columns=["temp", "mixing_time", "sem_porosity", "xrd_peak"],
        target_column="cycle_life",
        candidate_columns=["temp", "mixing_time"],
        pre_experiment_features=["temp", "mixing_time"],
        post_experiment_characterization=["sem_porosity", "xrd_peak"],
        targets=["cycle_life"],
        candidate_variables=["temp", "mixing_time"],
        oracle_columns=["hidden_pde_truth", "internal_defect_flag"],
    )


def test_experiment_record_valid_transitions() -> None:
    rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_A",
        dataset_name="battery_lab_v1",
        pre_experiment_features={"temp": 300.0, "mixing_time": 45.0},
    )
    assert rec.stage == ExperimentStage.PROPOSED
    assert not rec.is_terminal()

    # PROPOSED -> EXECUTED
    rec.transition_to(ExperimentStage.EXECUTED)
    assert rec.stage == ExperimentStage.EXECUTED

    # EXECUTED -> CHARACTERIZED
    rec.transition_to(
        ExperimentStage.CHARACTERIZED,
        characterization={"sem_porosity": 0.18, "xrd_peak": 42.5},
    )
    assert rec.stage == ExperimentStage.CHARACTERIZED
    assert rec.has_characterization()
    assert not rec.has_performance()

    # CHARACTERIZED -> PERFORMANCE_MEASURED
    rec.transition_to(
        ExperimentStage.PERFORMANCE_MEASURED,
        performance={"cycle_life": 920.0},
    )
    assert rec.stage == ExperimentStage.PERFORMANCE_MEASURED
    assert rec.has_performance()

    # PERFORMANCE_MEASURED -> COMPLETED
    rec.transition_to(ExperimentStage.COMPLETED)
    assert rec.stage == ExperimentStage.COMPLETED
    assert rec.is_terminal()


def test_experiment_record_copy() -> None:
    rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_A",
        dataset_name="battery_lab_v1",
        pre_experiment_features={"temp": 300.0},
        characterization={"sem_porosity": 0.18},
    )
    copied = rec.copy()
    assert copied.experiment_id == rec.experiment_id
    assert copied.pre_experiment_features == rec.pre_experiment_features

    # Mutating copy does not alter original
    copied.pre_experiment_features["temp"] = 500.0
    assert rec.pre_experiment_features["temp"] == 300.0


def test_experiment_record_invalid_transitions() -> None:
    rec = ScientificExperimentRecord(
        experiment_id="EXP_002",
        candidate_id="CAND_B",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.COMPLETED,
        pre_experiment_features={"temp": 300.0},
    )
    # Terminal stage cannot transition
    with pytest.raises(ValueError, match="Invalid stage transition"):
        rec.transition_to(ExperimentStage.PROPOSED)

    rec2 = ScientificExperimentRecord(
        experiment_id="EXP_003",
        candidate_id="CAND_C",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.PROPOSED,
    )
    # PROPOSED cannot jump directly to COMPLETED without EXECUTED
    with pytest.raises(ValueError, match="Invalid stage transition"):
        rec2.transition_to(ExperimentStage.COMPLETED)


def test_validation_oracle_leakage_rejection(sample_spec: DatasetSpec) -> None:
    # Record attempting to include hidden oracle column in pre_experiment_features
    leaky_rec = ScientificExperimentRecord(
        experiment_id="EXP_LEAK_01",
        candidate_id="CAND_L1",
        dataset_name="battery_lab_v1",
        pre_experiment_features={"temp": 300.0, "hidden_pde_truth": 1050.0},
    )
    with pytest.raises(InformationHorizonError, match="Oracle leakage detected"):
        validate_record_against_spec(leaky_rec, sample_spec)


def test_validation_proposal_horizon_enforcement(sample_spec: DatasetSpec) -> None:
    # Proposed record must NOT contain characterization or performance
    bad_proposal = ScientificExperimentRecord(
        experiment_id="EXP_PROP_01",
        candidate_id="CAND_P1",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.PROPOSED,
        pre_experiment_features={"temp": 300.0, "mixing_time": 30.0},
        characterization={"sem_porosity": 0.22},  # FORBIDDEN at proposal time
    )
    with pytest.raises(InformationHorizonError, match="Characterization measurements cannot be present at stage PROPOSED"):
        validate_record_against_spec(bad_proposal, sample_spec)

    bad_proposal_perf = ScientificExperimentRecord(
        experiment_id="EXP_PROP_02",
        candidate_id="CAND_P2",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.PROPOSED,
        pre_experiment_features={"temp": 300.0, "mixing_time": 30.0},
        performance={"cycle_life": 1000.0},  # FORBIDDEN at proposal time
    )
    with pytest.raises(InformationHorizonError, match="Performance outcomes cannot be present at stage PROPOSED"):
        validate_record_against_spec(bad_proposal_perf, sample_spec)


def test_validation_finite_values_and_completed_requirements(sample_spec: DatasetSpec) -> None:
    # Non-finite values rejected
    bad_val_rec = ScientificExperimentRecord(
        experiment_id="EXP_BAD_01",
        candidate_id="CAND_B1",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.PROPOSED,
        pre_experiment_features={"temp": float("nan"), "mixing_time": 30.0},
    )
    with pytest.raises(InformationHorizonError, match="non-finite numerical value"):
        validate_record_against_spec(bad_val_rec, sample_spec)

    # Completed stage requires primary target
    bad_completed = ScientificExperimentRecord(
        experiment_id="EXP_BAD_02",
        candidate_id="CAND_B2",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.COMPLETED,
        pre_experiment_features={"temp": 300.0, "mixing_time": 30.0},
        performance={},  # Missing "cycle_life"
    )
    with pytest.raises(InformationHorizonError, match="primary target 'cycle_life' is missing"):
        validate_record_against_spec(bad_completed, sample_spec)


def test_validation_valid_lifecycle_passes(sample_spec: DatasetSpec) -> None:
    # 1. Valid proposal
    rec = ScientificExperimentRecord(
        experiment_id="EXP_OK_01",
        candidate_id="CAND_OK1",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.PROPOSED,
        pre_experiment_features={"temp": 300.0, "mixing_time": 30.0},
        candidate_variables={"temp": 300.0, "mixing_time": 30.0},
    )
    validate_record_against_spec(rec, sample_spec)

    # 2. Executed -> Characterized
    rec.transition_to(ExperimentStage.EXECUTED)
    validate_record_against_spec(rec, sample_spec)

    rec.transition_to(
        ExperimentStage.CHARACTERIZED,
        characterization={"sem_porosity": 0.15, "xrd_peak": 44.0},
    )
    validate_record_against_spec(rec, sample_spec)

    # 3. Completed with performance
    rec.transition_to(
        ExperimentStage.COMPLETED,
        performance={"cycle_life": 980.0},
    )
    validate_record_against_spec(rec, sample_spec)


def test_record_does_not_infer_all_pre_features_as_controllable() -> None:
    rec = ScientificExperimentRecord(
        experiment_id="EXP_SUBSET_01",
        candidate_id="CAND_S1",
        dataset_name="battery_lab_v1",
        pre_experiment_features={"material_code": 101, "temp": 300.0},
        candidate_variables={"temp": 300.0},
    )
    assert "material_code" not in rec.candidate_variables
    assert "temp" in rec.candidate_variables
    assert rec.candidate_variables == {"temp": 300.0}


def test_partial_characterization_and_duplicate_measurement_detection() -> None:
    from src.science.records import DuplicateMeasurementError

    rec = ScientificExperimentRecord(
        experiment_id="EXP_PART_01",
        candidate_id="CAND_P1",
        dataset_name="battery_lab_v1",
        stage=ExperimentStage.EXECUTED,
        pre_experiment_features={"temp": 300.0},
    )

    assert not rec.has_any_characterization()
    assert not rec.has_required_characterization(["sem_porosity", "xrd_peak"])

    # 1. z1 arrives
    rec.transition_to(ExperimentStage.CHARACTERIZED, characterization={"sem_porosity": 0.18})
    assert rec.has_any_characterization()
    assert not rec.has_required_characterization(["sem_porosity", "xrd_peak"])

    # 2. Conflicting duplicate measurement without revision flag
    with pytest.raises(DuplicateMeasurementError, match="Duplicate characterization measurement"):
        rec.transition_to(ExperimentStage.CHARACTERIZED, characterization={"sem_porosity": 0.25})

    # 3. Valid additive measurement: z2 arrives
    rec.transition_to(ExperimentStage.CHARACTERIZED, characterization={"xrd_peak": 42.0})
    assert rec.has_required_characterization(["sem_porosity", "xrd_peak"])
    assert rec.characterization == {"sem_porosity": 0.18, "xrd_peak": 42.0}

    # 4. Explicit revision allowed
    rec.transition_to(
        ExperimentStage.CHARACTERIZED,
        characterization={"sem_porosity": 0.20},
        allow_measurement_revision=True,
    )
    assert rec.characterization["sem_porosity"] == 0.20

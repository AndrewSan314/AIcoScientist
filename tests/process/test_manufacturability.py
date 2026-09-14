import pandas as pd
import pytest

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.manufacturability import ManufacturabilityModel
from src.process.optimization.botorch import UnsupportedProcessOptimizationError
from src.process.optimization.process_objective import ConstraintSpec, ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from .test_scalar_constraints import _FirstFeasibleBackend, _observations


def _model():
    return ManufacturabilityModel(seed=7).fit(pd.DataFrame({"temperature": [90, 95, 100, 105, 115, 120, 125, 130]}), ["mixing_failure", "mixing_failure", "out_of_spec", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"])


def test_feasibility_probability_is_learned_and_filters_candidates():
    result = ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend(), manufacturability_model=_model()).propose_recipes(
        _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
        ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("manufacturing_success", "feasibility", 0.8)]),
    )
    assert result[0].controls == {"temperature": 120.0}
    assert result[0].feasibility_probability is not None and result[0].feasibility_probability >= 0.8
    assert result[0].provenance["feasibility_model_fingerprint"]


def test_feasibility_without_a_matching_model_fails_closed():
    with pytest.raises(UnsupportedProcessOptimizationError):
        ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend()).propose_recipes(
            _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
            ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("manufacturing_success", "feasibility", 0.8)]),
        )


def test_same_config_data_labels_produce_identical_fingerprints():
    df1 = pd.DataFrame({"temperature": [90.0, 95.0, 100.0, 105.0, 115.0, 120.0, 125.0, 130.0]})
    df2 = pd.DataFrame({"temperature": [90.0, 95.0, 100.0, 105.0, 115.0, 120.0, 125.0, 130.0]})
    labels = ["mixing_failure", "mixing_failure", "out_of_spec", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"]
    m1 = ManufacturabilityModel(seed=42).fit(df1, labels)
    m2 = ManufacturabilityModel(seed=42).fit(df2, labels)

    assert m1.training_data_fingerprint == m2.training_data_fingerprint
    assert m1.training_label_fingerprint == m2.training_label_fingerprint
    assert m1.fitted_state_fingerprint == m2.fitted_state_fingerprint
    assert m1.fingerprint == m2.fingerprint


def test_perturbing_feature_changes_data_and_overall_fingerprint():
    df_base = pd.DataFrame({"temperature": [90.0, 95.0, 100.0, 105.0, 115.0, 120.0, 125.0, 130.0]})
    df_perturbed = pd.DataFrame({"temperature": [90.0001, 95.0, 100.0, 105.0, 115.0, 120.0, 125.0, 130.0]})
    labels = ["mixing_failure", "mixing_failure", "out_of_spec", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"]
    m_base = ManufacturabilityModel(seed=42).fit(df_base, labels)
    m_perturbed = ManufacturabilityModel(seed=42).fit(df_perturbed, labels)

    assert m_base.training_data_fingerprint != m_perturbed.training_data_fingerprint
    assert m_base.training_label_fingerprint == m_perturbed.training_label_fingerprint
    assert m_base.fingerprint != m_perturbed.fingerprint


def test_flipping_label_changes_label_and_overall_fingerprint():
    df = pd.DataFrame({"temperature": [90.0, 95.0, 100.0, 105.0, 115.0, 120.0, 125.0, 130.0]})
    labels1 = ["mixing_failure", "mixing_failure", "out_of_spec", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"]
    labels2 = ["mixing_failure", "mixing_failure", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"]
    m1 = ManufacturabilityModel(seed=42).fit(df, labels1)
    m2 = ManufacturabilityModel(seed=42).fit(df, labels2)

    assert m1.training_data_fingerprint == m2.training_data_fingerprint
    assert m1.training_label_fingerprint != m2.training_label_fingerprint
    assert m1.fingerprint != m2.fingerprint


def test_changing_stage_or_label_name_changes_fingerprint():
    df = pd.DataFrame({"temperature": [90.0, 95.0, 100.0, 105.0, 115.0, 120.0, 125.0, 130.0]})
    labels = ["mixing_failure", "mixing_failure", "out_of_spec", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"]
    m1 = ManufacturabilityModel(supported_stage="MIXING", seed=42).fit(df, labels)
    m2 = ManufacturabilityModel(supported_stage="COATING", seed=42).fit(df, labels)
    m3 = ManufacturabilityModel(label_name="custom_feasibility", seed=42).fit(df, labels)

    assert m1.fingerprint != m2.fingerprint
    assert m1.fingerprint != m3.fingerprint


def test_coordinator_provenance_captures_feasibility_model_fingerprint():
    model = _model()
    result = ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend(), manufacturability_model=model).propose_recipes(
        _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
        ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("manufacturing_success", "feasibility", 0.8)]),
    )
    assert result[0].provenance["feasibility_model_fingerprint"] == model.fingerprint


def test_unfitted_model_raises_attribute_error_on_property_access():
    m = ManufacturabilityModel(seed=42)
    with pytest.raises(AttributeError, match="has not been fitted"):
        _ = m.fingerprint
    with pytest.raises(AttributeError, match="has not been fitted"):
        _ = m.training_data_fingerprint
    with pytest.raises(AttributeError, match="has not been fitted"):
        _ = m.training_label_fingerprint
    with pytest.raises(AttributeError, match="has not been fitted"):
        _ = m.fitted_state_fingerprint

    # But getattr with default returns the default
    assert getattr(m, "fingerprint", None) is None
    assert getattr(m, "training_data_fingerprint", None) is None

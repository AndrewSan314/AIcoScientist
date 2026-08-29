from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.base import DatasetAdapter, DatasetSpec
from src.datasets.severson import SeversonAdapter
from src.optimization.recommender import recommend


def test_prediction_only_dataset_spec():
    spec = DatasetSpec(
        name="test_pred_only",
        id_column="sample_id",
        feature_columns=["f1", "f2"],
        target_column="target",
        objective="maximize",
        supports_prediction=True,
        supports_optimization=False,
    )
    assert spec.supports_prediction is True
    assert spec.supports_optimization is False
    assert spec.candidate_columns == []
    assert spec.candidate_id_column is None


def test_optimization_spec_requires_candidate_columns():
    with pytest.raises(ValueError, match="candidate_columns must not be empty"):
        DatasetSpec(
            name="test_invalid_opt",
            id_column="sample_id",
            feature_columns=["f1", "f2"],
            target_column="target",
            objective="maximize",
            supports_optimization=True,
            candidate_columns=[],
        )



def test_severson_spec_capabilities():
    adapter = SeversonAdapter()
    assert adapter.spec.supports_prediction is True
    assert adapter.spec.supports_optimization is False
    assert adapter.spec.candidate_columns == []
    assert adapter.spec.candidate_id_column is None
    assert adapter.spec.entity_id_column == "physical_cell_id"


def test_severson_candidate_space_raises():
    adapter = SeversonAdapter()
    dummy_obs = pd.DataFrame()
    with pytest.raises(NotImplementedError, match="prediction-only dataset"):
        adapter.candidate_space(dummy_obs)


def test_recommender_rejects_prediction_only_dataset():
    adapter = SeversonAdapter()
    dummy_obs = pd.DataFrame(
        {
            "physical_cell_id": ["b1c0"],
            **{f: [1.0] for f in adapter.spec.feature_columns},
            "cycle_life": [1000.0],
        }
    )
    with pytest.raises(ValueError, match="prediction-only dataset and does not support recommendation"):
        recommend(adapter, dummy_obs)

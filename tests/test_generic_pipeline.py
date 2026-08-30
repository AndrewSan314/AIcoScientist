import numpy as np
import pandas as pd
import pytest

from src.build_dataset import build_dataset
from src.datasets.base import DatasetAdapter, DatasetSpec
from src.datasets.registry import get_dataset_adapter
from src.evaluation.metrics import best_seen, simple_regret, top_k_hit_rate
from src.evaluation.oracle import OfflineOracle
from src.evaluation.replay import replay
from src.legacy.native_optimizer.acquisition import ucb_acquisition as ucb
from src.optimization.candidates import remove_observed
from src.optimization.constraints import apply_constraints
from src.optimization.recommender import recommend
from src.train_model import train_model


class SyntheticAdapter(DatasetAdapter):
    def __init__(self, objective="maximize"):
        self._spec = DatasetSpec(
            name=f"synthetic_{objective}",
            id_column="sample_id",
            feature_columns=["x1", "x2"],
            target_column="target",
            objective=objective,
            candidate_columns=["x1", "x2"],
        )
        rows = []
        for x1 in range(5):
            for x2 in range(5):
                value = (x1 - 3) ** 2 + (x2 - 2) ** 2
                rows.append({
                    "sample_id": f"{x1}-{x2}",
                    "x1": x1,
                    "x2": x2,
                    "target": -value if objective == "maximize" else value,
                })
        self._df = pd.DataFrame(rows)

    @property
    def spec(self):
        return self._spec

    def load(self):
        return self._df.copy()

    def candidate_space(self, observed):
        return self._df[self.spec.candidate_columns].copy()


class EvenOnlyAdapter(SyntheticAdapter):
    def validate_candidate(self, candidate):
        if candidate["x1"] % 2:
            return False, ["x1 must be even"]
        return True, []


def test_dataset_spec_and_registry():
    with pytest.raises(ValueError, match="objective"):
        DatasetSpec("bad", "id", ["x"], "y", "sideways", ["x"])
    assert get_dataset_adapter("si_mxene").spec.name == "si_mxene"
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_adapter("missing")


def test_generic_build_and_train(tmp_path):
    adapter = SyntheticAdapter()
    df = build_dataset(adapter)
    assert len(df) == 25
    model_path = tmp_path / "model.pkl"
    metrics = train_model(df, adapter=adapter, output_path=model_path)
    assert {"rf_metrics", "gp_metrics"} <= set(metrics)
    bundle = __import__("joblib").load(model_path)
    assert bundle["dataset"] == "synthetic_maximize"
    assert bundle["features"] == ["x1", "x2"]
    assert bundle["target"] == "target"


def test_ucb_supports_both_objectives():
    np.testing.assert_allclose(ucb([1, 2], [0.5, 0.5]), [1.5, 2.5])
    np.testing.assert_allclose(
        ucb([1, 2], [0.5, 0.5], objective="minimize"), [-0.5, -1.5]
    )


def test_candidate_removal_and_constraints():
    adapter = EvenOnlyAdapter()
    observed = adapter.load().head(1)
    candidates = adapter.candidate_space(observed)
    remaining = remove_observed(candidates, observed, adapter.spec.candidate_columns)
    assert not ((remaining["x1"] == 0) & (remaining["x2"] == 0)).any()
    constrained = apply_constraints(remaining, adapter)
    assert constrained["x1"].mod(2).eq(0).all()


def test_generic_recommendation_and_metrics(tmp_path):
    adapter = SyntheticAdapter()
    observed = adapter.load().iloc[[0, 1, 5, 6]].copy()
    model_path = tmp_path / "model.pkl"
    train_model(observed, adapter=adapter, output_path=model_path)
    bundle = __import__("joblib").load(model_path)
    recommendations = recommend(adapter, observed, model_bundle=bundle, n=2)
    assert len(recommendations) == 2
    assert {"x1", "x2", "predicted_target", "acquisition_score"} <= set(recommendations)
    assert best_seen([1, 3, 2]) == 3
    assert best_seen([1, 3, 2], "minimize") == 1
    assert simple_regret(8, 10) == 2
    assert simple_regret(12, 10, "minimize") == 2
    assert top_k_hit_rate(["b", "x"], {"a": 3, "b": 2, "c": 1}, 2) == 0.5


def test_offline_oracle_and_replay(tmp_path):
    adapter = SyntheticAdapter()
    full = adapter.load()
    oracle = OfflineOracle(full, adapter.spec)
    answer = oracle.query({"x1": 3, "x2": 2})
    assert answer["target"] == 0
    with pytest.raises(KeyError, match="No exact"):
        oracle.query({"x1": 99, "x2": 99})

    ambiguous = pd.concat([full, full.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="ambiguous"):
        OfflineOracle(ambiguous, adapter.spec).query({"x1": 0, "x2": 0})

    initial = full.iloc[[0, 1, 5, 6]].copy()
    result = replay(adapter, oracle, initial, budget=1, model_dir=tmp_path)
    assert len(result["history"]) == 1
    assert len(result["observed"]) == len(initial) + 1
    assert best_seen(result["observed"]["target"]) >= best_seen(initial["target"])

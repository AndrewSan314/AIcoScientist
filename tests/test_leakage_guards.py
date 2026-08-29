import numpy as np
import pandas as pd
import pytest

from src.datasets.base import DatasetAdapter, DatasetBundle, DatasetSpec
from src.evaluation.oracle import OfflineOracle, OracleResponse
from src.evaluation.replay import replay
from src.train_model import make_train_test_split, train_model


class BaseTestAdapter(DatasetAdapter):
    def __init__(self, spec: DatasetSpec, df: pd.DataFrame):
        self._spec = spec
        self._df = df.copy()

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def load(self) -> pd.DataFrame:
        return self._df.copy()

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        return self._df[self.spec.candidate_columns].copy()


def test_1_oracle_feature_conflict():
    """1. ORACLE FEATURE CONFLICT: feature_columns intersecting oracle_columns must fail."""
    with pytest.raises(ValueError, match="overlap"):
        DatasetSpec(
            name="test_conflict",
            id_column="id",
            feature_columns=["x", "final_life"],
            target_column="capacity",
            objective="maximize",
            candidate_columns=["x"],
            oracle_columns=["final_life"],
        )


def test_2_target_as_feature():
    """2. TARGET AS FEATURE: target_column also present in feature_columns must fail."""
    with pytest.raises(ValueError, match="target_column"):
        DatasetSpec(
            name="test_target_leak",
            id_column="id",
            feature_columns=["x", "capacity"],
            target_column="capacity",
            objective="maximize",
            candidate_columns=["x"],
        )


def test_3_group_split_cell_isolation():
    """3. GROUP SPLIT: Multiple rows per physical cell never cross train/test boundaries."""
    rows = []
    for cell_id in ["cell_A", "cell_B", "cell_C", "cell_D", "cell_E", "cell_F"]:
        for cycle in [10, 20, 30]:
            rows.append({
                "sample_id": f"{cell_id}_{cycle}",
                "cell_id": cell_id,
                "x1": float(cycle),
                "target": float(100 - cycle * 0.5),
            })
    df = pd.DataFrame(rows)
    spec = DatasetSpec(
        name="cell_grouped",
        id_column="sample_id",
        entity_id_column="cell_id",
        feature_columns=["x1"],
        target_column="target",
        objective="maximize",
        candidate_columns=["x1"],
        split_group_columns=["cell_id"],
    )

    train_idx, test_idx = make_train_test_split(df, spec, test_size=0.33, random_state=42)
    train_cells = set(df.iloc[train_idx]["cell_id"])
    test_cells = set(df.iloc[test_idx]["cell_id"])

    assert len(train_cells) >= 1
    assert len(test_cells) >= 1
    assert not (train_cells & test_cells), f"Cells leaked across split: {train_cells & test_cells}"


def test_4_protocol_replicates_grouped_split():
    """4. PROTOCOL REPLICATES: Replicate rows for protocols stay grouped together."""
    rows = [
        {"sample_id": "s1", "protocol_id": "P1", "replicate": "A", "feat": 1.0, "target": 500.0},
        {"sample_id": "s2", "protocol_id": "P1", "replicate": "B", "feat": 1.0, "target": 520.0},
        {"sample_id": "s3", "protocol_id": "P2", "replicate": "A", "feat": 2.0, "target": 600.0},
        {"sample_id": "s4", "protocol_id": "P2", "replicate": "B", "feat": 2.0, "target": 610.0},
        {"sample_id": "s5", "protocol_id": "P3", "replicate": "A", "feat": 3.0, "target": 700.0},
        {"sample_id": "s6", "protocol_id": "P3", "replicate": "B", "feat": 3.0, "target": 705.0},
        {"sample_id": "s7", "protocol_id": "P4", "replicate": "A", "feat": 4.0, "target": 800.0},
        {"sample_id": "s8", "protocol_id": "P4", "replicate": "B", "feat": 4.0, "target": 815.0},
    ]
    df = pd.DataFrame(rows)
    spec = DatasetSpec(
        name="protocol_grouped",
        id_column="sample_id",
        candidate_id_column="protocol_id",
        feature_columns=["feat"],
        target_column="target",
        objective="maximize",
        candidate_columns=["feat"],
        split_group_columns=["protocol_id"],
    )

    train_idx, test_idx = make_train_test_split(df, spec, test_size=0.25, random_state=42)
    train_protocols = set(df.iloc[train_idx]["protocol_id"])
    test_protocols = set(df.iloc[test_idx]["protocol_id"])

    assert not (train_protocols & test_protocols), "Protocols leaked across train and test"
    # Ensure all rows for each protocol are together
    for protocol in ["P1", "P2", "P3", "P4"]:
        indices = df.index[df["protocol_id"] == protocol].tolist()
        in_train = [i in train_idx for i in indices]
        assert all(in_train) or not any(in_train), f"Protocol {protocol} was split across train/test"


def test_5_oracle_raw_data_is_hidden():
    """5. ORACLE RAW DATA IS HIDDEN: Oracle query result must not expose secret/raw columns."""
    df = pd.DataFrame([
        {
            "sample_id": "c1",
            "x1": 10,
            "x2": 20,
            "target": 95.0,
            "secret_eol_metric": "INTERNAL_GROUND_TRUTH_123",
            "future_resistance": 0.045,
        }
    ])
    spec = DatasetSpec(
        name="test_oracle_hide",
        id_column="sample_id",
        feature_columns=["x1", "x2"],
        target_column="target",
        objective="maximize",
        candidate_columns=["x1", "x2"],
        oracle_columns=["secret_eol_metric", "future_resistance"],
    )
    oracle = OfflineOracle(df, spec)
    response = oracle.query({"x1": 10, "x2": 20})

    assert isinstance(response, OracleResponse)
    assert response.target == 95.0
    assert response.candidate == {"x1": 10, "x2": 20}
    assert response.observations == {}
    assert "secret_eol_metric" not in response.candidate
    assert "secret_eol_metric" not in response.observations
    assert "future_resistance" not in response.observations

    with pytest.raises(KeyError, match="not accessible"):
        _ = response["secret_eol_metric"]

    with pytest.raises(KeyError, match="not accessible"):
        _ = response["row"]


def test_6_replicate_aggregation_mean():
    """6. REPLICATE AGGREGATION: replicate_policy='mean' returns mean target, n_replicates, target_std."""
    df = pd.DataFrame([
        {"sample_id": "r1", "protocol": "P17", "target": 10.0},
        {"sample_id": "r2", "protocol": "P17", "target": 14.0},
    ])
    spec = DatasetSpec(
        name="test_replicates",
        id_column="sample_id",
        candidate_id_column="protocol",
        feature_columns=["protocol"],
        target_column="target",
        objective="maximize",
        candidate_columns=["protocol"],
    )
    oracle = OfflineOracle(df, spec, replicate_policy="mean")
    response = oracle.query({"protocol": "P17"})

    assert response.target == 12.0
    assert response.metadata["n_replicates"] == 2
    assert response.metadata["target_std"] == pytest.approx(2.8284, rel=1e-3)


def test_7_ambiguous_replicate_with_error_policy():
    """7. AMBIGUOUS REPLICATE WITH ERROR POLICY: replicate_policy='error' raises ValueError."""
    df = pd.DataFrame([
        {"sample_id": "r1", "protocol": "P17", "target": 10.0},
        {"sample_id": "r2", "protocol": "P17", "target": 14.0},
    ])
    spec = DatasetSpec(
        name="test_replicates_error",
        id_column="sample_id",
        candidate_id_column="protocol",
        feature_columns=["protocol"],
        target_column="target",
        objective="maximize",
        candidate_columns=["protocol"],
    )
    oracle = OfflineOracle(df, spec, replicate_policy="error")
    with pytest.raises(ValueError, match="ambiguous"):
        oracle.query({"protocol": "P17"})


def test_8_unsafe_feature_construction_raises():
    """8. UNSAFE FEATURE CONSTRUCTION: Base adapter requiring missing feature raises ValueError."""
    spec = DatasetSpec(
        name="test_strict_features",
        id_column="sample_id",
        feature_columns=["x1", "secret_future_feature"],
        target_column="target",
        objective="maximize",
        candidate_columns=["x1"],
    )
    df = pd.DataFrame([
        {"sample_id": "s1", "x1": 1.0, "secret_future_feature": 5.0, "target": 10.0},
        {"sample_id": "s2", "x1": 2.0, "secret_future_feature": 6.0, "target": 20.0},
    ])
    adapter = BaseTestAdapter(spec, df)
    candidates = pd.DataFrame([{"x1": 3.0}])
    observed = df.copy()
    fill_values = {"secret_future_feature": 5.5}

    with pytest.raises(ValueError, match="Cannot construct pre-experiment feature"):
        adapter.build_candidate_features(candidates, observed, fill_values)


def test_9_replay_leakage_prevention(tmp_path):
    """9. REPLAY LEAKAGE: Full oracle row secrets never leak into observed DataFrame after replay step."""
    df = pd.DataFrame([
        {"sample_id": "s1", "x1": 1, "target": 10.0, "secret_future_feature": "SECRET_A"},
        {"sample_id": "s2", "x1": 2, "target": 20.0, "secret_future_feature": "SECRET_B"},
        {"sample_id": "s3", "x1": 3, "target": 30.0, "secret_future_feature": "SECRET_C"},
        {"sample_id": "s4", "x1": 4, "target": 40.0, "secret_future_feature": "SECRET_D"},
        {"sample_id": "s5", "x1": 5, "target": 50.0, "secret_future_feature": "SECRET_E"},
    ])
    spec = DatasetSpec(
        name="test_replay_leak",
        id_column="sample_id",
        feature_columns=["x1"],
        target_column="target",
        objective="maximize",
        candidate_columns=["x1"],
        oracle_columns=["secret_future_feature"],
    )
    adapter = BaseTestAdapter(spec, df)
    oracle = OfflineOracle(df, spec)

    initial_observed = df.drop(columns=["secret_future_feature"]).head(4).copy()
    assert "secret_future_feature" not in initial_observed.columns

    result = replay(adapter, oracle, initial_observed, budget=1, model_dir=tmp_path)
    observed = result["observed"]

    assert len(observed) == 5
    assert "secret_future_feature" not in observed.columns, "Oracle secret leaked into observed data!"
    assert set(observed.columns) == {"sample_id", "x1", "target"}


def test_10_minimum_split_sanity_and_group_checks():
    """11. MINIMUM EVALUATION SPLIT SANITY: Fail fast on invalid sizes and group counts."""
    spec_grouped = DatasetSpec(
        name="test_group_min",
        id_column="id",
        feature_columns=["x"],
        target_column="y",
        objective="maximize",
        candidate_columns=["x"],
        split_group_columns=["group"],
    )

    # 1 unique group must fail
    df_one_group = pd.DataFrame([
        {"id": "1", "group": "g1", "x": 1.0, "y": 10.0},
        {"id": "2", "group": "g1", "x": 2.0, "y": 20.0},
        {"id": "3", "group": "g1", "x": 3.0, "y": 30.0},
        {"id": "4", "group": "g1", "x": 4.0, "y": 40.0},
    ])
    with pytest.raises(ValueError, match="At least 2 unique groups"):
        make_train_test_split(df_one_group, spec_grouped)

    # < 4 total rows must fail
    df_small = pd.DataFrame([
        {"id": "1", "x": 1.0, "y": 10.0},
        {"id": "2", "x": 2.0, "y": 20.0},
        {"id": "3", "x": 3.0, "y": 30.0},
    ])
    spec_simple = DatasetSpec(
        name="test_simple_min",
        id_column="id",
        feature_columns=["x"],
        target_column="y",
        objective="maximize",
        candidate_columns=["x"],
    )
    with pytest.raises(ValueError, match="At least 4 rows"):
        make_train_test_split(df_small, spec_simple)


def test_dataset_spec_extended_validation():
    """Test extended DatasetSpec validation rules (duplicates, blanks, etc.)."""
    # Empty column name in split_group_columns
    with pytest.raises(ValueError, match="split_group_columns"):
        DatasetSpec("d", "id", ["x"], "y", "maximize", ["x"], split_group_columns=[""])

    # Duplicate in split_group_columns
    with pytest.raises(ValueError, match="split_group_columns"):
        DatasetSpec("d", "id", ["x"], "y", "maximize", ["x"], split_group_columns=["grp", "grp"])

    # Duplicate in oracle_columns
    with pytest.raises(ValueError, match="oracle_columns"):
        DatasetSpec("d", "id", ["x"], "y", "maximize", ["x"], oracle_columns=["orc", "orc"])

    # Duplicate in observation_columns
    with pytest.raises(ValueError, match="observation_columns"):
        DatasetSpec("d", "id", ["x"], "y", "maximize", ["x"], observation_columns=["obs", "obs"])


def test_dataset_bundle():
    """Verify DatasetBundle representation and semantics."""
    candidates = pd.DataFrame([{"x": 1}, {"x": 2}])
    observations = pd.DataFrame([{"id": "s1", "x": 1, "target": 10.0}])
    oracle = pd.DataFrame([{"id": "s1", "x": 1, "target": 10.0, "secret": 99}])
    provenance = {"dataset": "synthetic", "version": "1.0"}

    bundle = DatasetBundle(
        candidates=candidates,
        observations=observations,
        oracle=oracle,
        provenance=provenance,
    )
    assert len(bundle.candidates) == 2
    assert len(bundle.observations) == 1
    assert "secret" in bundle.oracle.columns
    assert "secret" not in bundle.observations.columns
    assert bundle.provenance["version"] == "1.0"

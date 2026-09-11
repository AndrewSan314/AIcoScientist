from __future__ import annotations

import pandas as pd
import numpy as np

from src.optimization.finite_pool import FiniteCandidatePool
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.optimization.state import (
    canonical_control_action_id,
    context_provenance_fingerprint,
    contextual_candidate_instance_id,
)
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage


def test_process_space_proposes_only_exact_recorded_recipes() -> None:
    space = ProcessSearchSpace.from_finite_pool(pd.DataFrame([{"recipe_id": "a", "temperature": 100}, {"recipe_id": "b", "temperature": 120}]))
    assert space.validate_recipe({"temperature": 100})
    assert not space.validate_recipe({"temperature": 110})


def test_official_process_scaling_uses_known_controls_without_changing_recipe_ids() -> None:
    space = ProcessSearchSpace(pd.DataFrame({"recipe_id": ["a", "b"], "speed": [10.0, 20.0], "gap": [0.1, 0.1]}))
    observations = pd.DataFrame({"recipe_id": ["a"], "speed": [10.0], "gap": [0.1], "target": [1.0]})

    scaled_observations, scaled_space = ProcessOptimizationCoordinator._scale_official_botorch_inputs(observations, space)

    assert scaled_space.candidates["recipe_id"].tolist() == ["a", "b"]
    assert scaled_space.candidates["control_0_value"].tolist() == [0.0, 1.0]
    assert scaled_space.candidates["control_1_value"].tolist() == [0.0, 0.0]
    assert scaled_observations.loc[0, "target"] == 1.0


def test_official_process_view_one_hot_encodes_only_source_categories() -> None:
    space = ProcessSearchSpace(pd.DataFrame({"recipe_id": ["a", "b"], "protocol": ["fast", "slow"]}))
    observations, encoded_space = space.official_botorch_view(pd.DataFrame({"recipe_id": ["a"], "protocol": ["fast"], "target": [1.0]}))

    assert encoded_space.candidates[["control_0_category_0", "control_0_category_1"]].values.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert observations.loc[0, "target"] == 1.0


def test_official_process_view_keeps_a_real_missing_sentinel_category_distinct_from_absence() -> None:
    space = ProcessSearchSpace(pd.DataFrame({"recipe_id": ["a", "b"], "protocol": ["__MISSING__", None]}))
    _, encoded_space = space.official_botorch_view(pd.DataFrame({"recipe_id": ["a"], "protocol": ["__MISSING__"], "target": [1.0]}))

    assert encoded_space.candidates[["control_0_category_0", "control_0_category_1"]].values.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_contextual_process_view_scales_state_separately_from_controls() -> None:
    space = ProcessSearchSpace.from_finite_pool(
        pd.DataFrame({"recipe_id": ["a", "b"], "temperature": [100.0, 120.0], "state": [0.0, 10.0]}),
        context_columns=("state",), context_bounds={"state": (0.0, 10.0)},
    )
    observations, encoded_space = space.official_botorch_view(
        pd.DataFrame({"recipe_id": ["a"], "temperature": [100.0], "state": [5.0], "target": [1.0]})
    )
    assert encoded_space.model_columns == ["state", "control_0_value"]
    assert encoded_space.candidates["state"].tolist() == [0.0, 1.0]
    assert observations["state"].tolist() == [0.5]
    assert space.recipe("b") == {"temperature": 120.0}
    assert encoded_space.recipe("b") == {"control_0_value": 1.0}


def test_contextual_identity_reuses_actions_only_with_the_same_context_and_stage() -> None:
    action_a = canonical_control_action_id({"temperature": 100})
    action_b = canonical_control_action_id({"temperature": 120})
    historical_a = contextual_candidate_instance_id("context-02", action_a, ProcessStage.DRYING)
    current_a = contextual_candidate_instance_id("context-08", action_a, ProcessStage.DRYING)
    current_b = contextual_candidate_instance_id("context-08", action_b, ProcessStage.DRYING)

    assert action_a == canonical_control_action_id({"temperature": 100})
    assert historical_a != current_a
    assert current_a != current_b
    assert {current_a, current_b} - {historical_a} == {current_a, current_b}
    assert {current_a, current_b} - {current_a} == {current_b}


def test_generic_finite_candidate_pool_identity_semantics_remain_unchanged() -> None:
    pool = FiniteCandidatePool(
        pd.DataFrame({"candidate_id": ["a", "b"], "temperature": [100.0, 120.0]}),
        feature_columns=["temperature"], id_column="candidate_id",
    )

    assert pool.filter_unseen({"a"}).candidate_ids == ["b"]


def test_action_identity_canonicalizes_all_project_missing_scalars_without_confusing_zero_or_category() -> None:
    missing_ids = [canonical_control_action_id({"temperature": value}) for value in (None, float("nan"), np.float64(np.nan), pd.NA)]

    assert len(set(missing_ids)) == 1
    assert missing_ids[0] != canonical_control_action_id({"temperature": 0.0})
    assert missing_ids[0] != canonical_control_action_id({"temperature": "__MISSING__"})
    assert canonical_control_action_id({"a": 1, "b": 2}) == canonical_control_action_id({"b": 2, "a": 1})


def test_context_fingerprint_carries_semantic_model_and_category_identity() -> None:
    base = {"context.state.0": 0.25}
    first = context_provenance_fingerprint(
        base, ProcessStage.DRYING, "multimodal_stage_state",
        semantic_metadata={"source_stage_ids": ("form", "mix"), "model_version": "v1", "model_fingerprint": "a", "modality_bindings": {"signal": "s", "tabular": "t"}, "category_vocabulary_manifest": {"mode": ("fast", "slow")}},
    )
    reordered = context_provenance_fingerprint(
        base, ProcessStage.DRYING, "multimodal_stage_state",
        semantic_metadata={"category_vocabulary_manifest": {"mode": ("fast", "slow")}, "modality_bindings": {"tabular": "t", "signal": "s"}, "model_fingerprint": "a", "model_version": "v1", "source_stage_ids": ("form", "mix")},
    )
    other_model = context_provenance_fingerprint(
        base, ProcessStage.DRYING, "multimodal_stage_state",
        semantic_metadata={"source_stage_ids": ("form", "mix"), "model_version": "v2", "model_fingerprint": "b"},
    )
    other_vocabulary = context_provenance_fingerprint(
        base, ProcessStage.DRYING, "scalar_horizon",
        semantic_metadata={"source_stage_ids": ("form", "mix"), "category_vocabulary_manifest": {"mode": ("fast", "slow", "turbo")}},
    )

    assert first == reordered
    assert first != other_model
    assert first != other_vocabulary
    assert first == context_provenance_fingerprint(
        base, ProcessStage.DRYING, "multimodal_stage_state",
        semantic_metadata={"source_stage_ids": ("other-form", "other-mix"), "model_version": "v1", "model_fingerprint": "a", "modality_bindings": {"signal": "s", "tabular": "t"}, "category_vocabulary_manifest": {"mode": ("fast", "slow")}},
    )
    assert context_provenance_fingerprint(
        base, ProcessStage.DRYING, "multimodal_stage_state",
        semantic_metadata={"source_stage_ids": ("form", "mix"), "timestamp": "2026-09-11T00:00:00Z"},
    ) == context_provenance_fingerprint(
        base, ProcessStage.DRYING, "multimodal_stage_state",
        semantic_metadata={"source_stage_ids": ("form", "mix"), "timestamp": "2026-09-12T00:00:00Z"},
    )

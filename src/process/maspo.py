from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from numbers import Real
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import BatteryProcessRun
from .coordinator import ProcessOptimizationCoordinator
from .information_horizon import HorizonView
from .optimization.process_objective import ProcessOptimizationObjective
from .optimization.process_space import ProcessSearchSpace
from .optimization.proposal import ProcessControlProposal
from .optimization.state import (
    ModelValidationStatus,
    OptimizationState,
    canonical_control_action_id,
    contextual_candidate_instance_id,
    context_provenance_fingerprint,
)
from .stages import ProcessStage, STAGE_ORDER


SOURCE_RECIPE_ID_COLUMN = "source_recipe_id"
SOURCE_RECIPE_IDS_COLUMN = "source_recipe_ids"
CONTROL_ACTION_ID_COLUMN = "control_action_id"
CANDIDATE_INSTANCE_ID_COLUMN = "candidate_instance_id"
CONTEXT_PROVENANCE_COLUMN = "context_provenance_fingerprint"


def _numeric_context(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("legal process context contains a non-finite numeric value")
    return numeric


def _category_key(value: object) -> tuple[str, str]:
    try:
        if bool(pd.isna(value)):
            return ("missing", "")
    except (TypeError, ValueError):
        pass
    return ("value", str(value))


def _horizon_context_items(horizon: HorizonView) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for prefix, source in (("context.control.", horizon.controls), ("context.intermediate.", horizon.intermediate_properties)):
        items.extend((prefix + name, item.value) for name, item in source.items())
    return items


@dataclass(frozen=True)
class MASPOPlan:
    current_stage: ProcessStage
    next_stage: ProcessStage
    next_control: ProcessControlProposal
    downstream_stages: tuple[ProcessStage, ...]
    legal_state: HorizonView
    optimization_state: OptimizationState
    decision_latency_seconds: float


@dataclass(frozen=True)
class ContextualProcessState(OptimizationState):
    """Scalar state encoded from one legal InformationHorizon view."""

    context_columns: tuple[str, ...] = ()
    context_feature_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_horizon(
        cls,
        horizon: HorizonView,
        *,
        categorical_vocabularies: Mapping[str, Sequence[object]] | None = None,
    ) -> "ContextualProcessState":
        values: dict[str, float] = {}
        feature_map: dict[str, tuple[str, ...]] = {}
        category_manifest: dict[str, tuple[tuple[str, str], ...]] = {}
        context_columns = tuple(name for name, _ in _horizon_context_items(horizon))
        for raw_name, raw_value in _horizon_context_items(horizon):
            numeric = _numeric_context(raw_value)
            if numeric is not None:
                values[raw_name] = numeric
                feature_map[raw_name] = (raw_name,)
                continue
            vocabulary = None if categorical_vocabularies is None else categorical_vocabularies.get(raw_name)
            if vocabulary is None:
                raise ValueError(f"categorical legal context {raw_name!r} requires a source-supported vocabulary")
            categories = tuple(sorted({_category_key(item) for item in vocabulary}))
            current = _category_key(raw_value)
            if not categories or current not in categories:
                raise ValueError(f"categorical legal context {raw_name!r} is unknown to the source vocabulary")
            features = tuple(f"{raw_name}.category.{index}" for index in range(len(categories)))
            feature_map[raw_name] = features
            category_manifest[raw_name] = categories
            values.update({feature: float(category == current) for feature, category in zip(features, categories)})
        if not values:
            raise ValueError("contextual MASPO requires a finite legal HorizonView feature; no zero-state fallback")
        semantic_metadata = {
            "category_vocabulary_manifest": category_manifest,
        }
        fingerprint = context_provenance_fingerprint(
            values, horizon.decision_stage, "scalar_horizon", semantic_metadata=semantic_metadata,
        )
        return cls(
            feature_names=tuple(values), feature_values=values, decision_stage=horizon.decision_stage,
            source_stage_ids=horizon.source_stage_ids, provenance_fingerprint=fingerprint,
            representation_kind="scalar_horizon", provenance={
                "category_vocabularies": category_manifest,
                "semantic_fingerprint_inputs": semantic_metadata,
                "audit_provenance": {"source_stage_ids": horizon.source_stage_ids},
            },
            context_columns=context_columns, context_feature_map=feature_map,
        )


class MASPOProcessOptimizationCoordinator:
    """Stage-wise contextual receding-horizon wrapper around the official optimizer."""

    def __init__(self, optimizer: ProcessOptimizationCoordinator | None = None, *, require_validated_multimodal_state: bool = True) -> None:
        self.optimizer = optimizer or ProcessOptimizationCoordinator()
        self.require_validated_multimodal_state = require_validated_multimodal_state

    def optimize_remaining_process(
        self,
        *,
        current_state: BatteryProcessRun | HorizonView,
        current_stage: ProcessStage,
        remaining_control_spaces: dict[ProcessStage, ProcessSearchSpace],
        observations: pd.DataFrame | dict[ProcessStage, pd.DataFrame],
        objective: ProcessOptimizationObjective,
        seed: int | None = None,
        optimization_state: OptimizationState | None = None,
        historical_latent_metadata: Mapping[str, Any] | None = None,
    ) -> MASPOPlan:
        started = perf_counter()
        legal_state = (
            self.optimizer.available_state(current_state, current_stage)
            if isinstance(current_state, BatteryProcessRun)
            else current_state
        )
        if legal_state.decision_stage != current_stage:
            raise ValueError("current HorizonView decision_stage must match current_stage")
        if (
            optimization_state is not None
            and optimization_state.representation_kind in {"multimodal_stage_state", "scalar_plus_multimodal_stage_state", "multimodal_fused_baseline"}
            and self.require_validated_multimodal_state
            and optimization_state.validation_status != ModelValidationStatus.SOURCE_BACKED_VALIDATED
        ):
            raise ValueError("production MASPO requires SOURCE_BACKED_VALIDATED multimodal optimization state")
        stages = tuple(sorted((stage for stage in remaining_control_spaces if STAGE_ORDER[stage] > STAGE_ORDER[current_stage]), key=STAGE_ORDER.__getitem__))
        if not stages:
            raise ValueError("no remaining process control stage exists after current_stage")
        next_stage = stages[0]
        stage_observations = observations[next_stage] if isinstance(observations, dict) else observations
        state, contextual_observations, contextual_space = self._contextual_inputs(
            legal_state, remaining_control_spaces[next_stage], stage_observations, objective, optimization_state=optimization_state,
            historical_latent_metadata=historical_latent_metadata,
        )
        proposals = self.optimizer.propose_recipes(contextual_observations, contextual_space, objective, n=1, seed=seed)
        if not proposals:
            raise ValueError("official process optimizer returned no next-stage action")
        action = replace(proposals[0], stage=next_stage)
        return MASPOPlan(
            current_stage=current_stage, next_stage=next_stage, next_control=action,
            downstream_stages=stages[1:], legal_state=legal_state, optimization_state=state,
            decision_latency_seconds=perf_counter() - started,
        )

    @staticmethod
    def _contextual_inputs(
        legal_state: HorizonView,
        space: ProcessSearchSpace,
        observations: pd.DataFrame,
        objective: ProcessOptimizationObjective,
        *,
        optimization_state: OptimizationState | None = None,
        historical_latent_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[OptimizationState, pd.DataFrame, ProcessSearchSpace]:
        if not isinstance(observations, pd.DataFrame):
            raise TypeError("contextual MASPO observations must be a pandas DataFrame")
        state = optimization_state or ContextualProcessState.from_horizon(
            legal_state, categorical_vocabularies=MASPOProcessOptimizationCoordinator._categorical_vocabularies(legal_state, observations),
        )
        if state.decision_stage != legal_state.decision_stage:
            raise ValueError("optimization state decision_stage must match the legal HorizonView")
        if state.representation_kind != "scalar_horizon":
            metadata = historical_latent_metadata or observations.attrs.get("latent_artifact_metadata") or _latent_metadata_from_rows(observations)
            _validate_historical_latent_provenance(state, metadata)
        if space.context_columns:
            raise ValueError("MASPO expects an uncontextualized finite control space")
        context_columns = tuple(getattr(state, "context_columns", ())) or state.feature_names
        target_names = tuple(item.target for item in objective.objectives)
        source_column = space.source_id_column
        required = list(dict.fromkeys([space.id_column, source_column, *space.control_columns, *context_columns, *target_names]))
        missing = [column for column in required if column not in observations]
        if missing:
            raise ValueError(f"contextual MASPO requires source-backed historical context columns: {missing}")
        if observations.empty or observations[source_column].isna().any():
            raise ValueError("contextual MASPO requires non-null historical recipe identities")
        known_ids = set(space.candidates[source_column].astype(str))
        unknown_ids = sorted(set(observations[source_column].astype(str)) - known_ids)
        if unknown_ids:
            raise ValueError(f"historical observations contain unknown source recipes: {unknown_ids}")
        base = observations.loc[:, required].copy()
        history_context, context_bounds = MASPOProcessOptimizationCoordinator._encode_historical_context(base, state, context_columns)
        for name in target_names:
            values = pd.to_numeric(base[name], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"historical target column {name!r} is missing or non-finite")

        source_pool = space.candidates.copy()
        all_action_ids = [canonical_control_action_id(row[space.control_columns].to_dict()) for _, row in source_pool.iterrows()]
        action_groups: dict[str, list[int]] = {}
        for index, action_id in enumerate(all_action_ids):
            action_groups.setdefault(action_id, []).append(index)
        candidate_indices = [indices[0] for indices in action_groups.values()]
        action_ids = list(action_groups)
        candidate_pool = source_pool.iloc[candidate_indices].copy().reset_index(drop=True)
        candidate_pool[SOURCE_RECIPE_ID_COLUMN] = candidate_pool[source_column]
        candidate_pool[SOURCE_RECIPE_IDS_COLUMN] = [
            [str(source_pool.iloc[index][source_column]) for index in indices]
            for indices in action_groups.values()
        ]
        candidate_pool[CONTROL_ACTION_ID_COLUMN] = action_ids
        candidate_pool[CONTEXT_PROVENANCE_COLUMN] = state.provenance_fingerprint
        candidate_pool[CANDIDATE_INSTANCE_ID_COLUMN] = [
            contextual_candidate_instance_id(state.provenance_fingerprint, action_id, legal_state.decision_stage)
            for action_id in action_ids
        ]
        for name, value in state.feature_values.items():
            candidate_pool[name] = value
        if candidate_pool[CANDIDATE_INSTANCE_ID_COLUMN].duplicated().any():
            raise ValueError("finite process pool contains duplicate control actions under contextual identity")

        historical_action_ids = [canonical_control_action_id(row[space.control_columns].to_dict()) for _, row in base.iterrows()]
        historical_context_fingerprints = [
            context_provenance_fingerprint(
                history_context.iloc[index].to_dict(), state.decision_stage, state.representation_kind,
                semantic_metadata=MASPOProcessOptimizationCoordinator._state_semantic_metadata(state),
            )
            for index in range(len(history_context))
        ]
        clean = pd.DataFrame({
            CANDIDATE_INSTANCE_ID_COLUMN: [
                contextual_candidate_instance_id(fingerprint, action_id, state.decision_stage)
                for fingerprint, action_id in zip(historical_context_fingerprints, historical_action_ids)
            ],
            SOURCE_RECIPE_ID_COLUMN: base[source_column].tolist(),
            CONTROL_ACTION_ID_COLUMN: historical_action_ids,
            CONTEXT_PROVENANCE_COLUMN: historical_context_fingerprints,
        }, index=base.index)
        for column in state.feature_names:
            clean[column] = history_context[column]
        for column in space.control_columns:
            clean[column] = base[column]
        for column in target_names:
            clean[column] = base[column]
        metadata_columns = tuple(dict.fromkeys([
            space.id_column, source_column, SOURCE_RECIPE_ID_COLUMN, SOURCE_RECIPE_IDS_COLUMN,
            CONTROL_ACTION_ID_COLUMN, CONTEXT_PROVENANCE_COLUMN,
            *space.metadata_columns,
        ]))
        contextual_space = ProcessSearchSpace.from_finite_pool(
            candidate_pool, id_column=CANDIDATE_INSTANCE_ID_COLUMN, context_columns=state.feature_names,
            context_bounds=context_bounds, source_recipe_id_column=SOURCE_RECIPE_ID_COLUMN,
            control_action_id_column=CONTROL_ACTION_ID_COLUMN, metadata_columns=metadata_columns,
        )
        return state, clean, contextual_space

    @staticmethod
    def _state_semantic_metadata(state: OptimizationState) -> Mapping[str, Any]:
        return dict(state.provenance.get("semantic_fingerprint_inputs", {}))

    @staticmethod
    def _categorical_vocabularies(horizon: HorizonView, observations: pd.DataFrame) -> dict[str, Sequence[object]]:
        return {
            name: observations[name].tolist()
            for name, value in _horizon_context_items(horizon)
            if _numeric_context(value) is None and name in observations
        }

    @staticmethod
    def _encode_historical_context(
        base: pd.DataFrame, state: OptimizationState, context_columns: tuple[str, ...],
    ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
        context_map = getattr(state, "context_feature_map", {})
        category_vocabularies = getattr(state, "provenance", {}).get("category_vocabularies", {})
        encoded = pd.DataFrame(index=base.index)
        bounds: dict[str, tuple[float, float]] = {}
        if isinstance(state, ContextualProcessState):
            for raw_name in context_columns:
                feature_names = context_map[raw_name]
                if len(feature_names) == 1 and feature_names[0] == raw_name:
                    values = pd.to_numeric(base[raw_name], errors="coerce")
                    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                        raise ValueError(f"historical context column {raw_name!r} is missing or non-finite")
                    current = state.feature_values[raw_name]
                    lower, upper = float(values.min()), float(values.max())
                    if current < lower or current > upper:
                        raise ValueError(f"current legal context {raw_name!r} lies outside historical support")
                    encoded[raw_name] = values.astype(float)
                    bounds[raw_name] = (lower, upper)
                    continue
                categories = tuple(tuple(item) for item in category_vocabularies.get(raw_name, ()))
                if not categories:
                    raise ValueError(f"categorical historical context {raw_name!r} lacks a source vocabulary")
                values = base[raw_name].map(_category_key)
                unknown = sorted(set(values) - set(categories))
                if unknown:
                    raise ValueError(f"historical context {raw_name!r} contains unknown categories: {unknown}")
                for feature, category in zip(feature_names, categories):
                    encoded[feature] = (values == category).astype(float)
                    bounds[feature] = (0.0, 1.0)
        else:
            for name in state.feature_names:
                if name not in base:
                    raise ValueError(f"contextual MASPO requires historical latent context column: {name!r}")
                values = pd.to_numeric(base[name], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                    raise ValueError(f"historical latent context column {name!r} is missing or non-finite")
                current = state.feature_values[name]
                lower, upper = float(values.min()), float(values.max())
                if current < lower or current > upper:
                    raise ValueError(f"current latent context {name!r} lies outside historical support")
                encoded[name] = values.astype(float)
                bounds[name] = (lower, upper)
        return encoded, bounds


def _validate_historical_latent_provenance(
    state: OptimizationState, metadata: Mapping[str, Any] | None,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("historical multimodal latent rows require an artifact provenance manifest")
    current = state.provenance.get("semantic_fingerprint_inputs", {})
    expected = {
        "representation_kind": state.representation_kind,
        "model_fingerprint": state.provenance.get("model_fingerprint") or current.get("model_fingerprint"),
        "encoder_fingerprint": current.get("encoder_fingerprint"),
        "model_version": state.provenance.get("model_version") or current.get("model_version"),
        "latent_feature_schema": tuple(state.feature_names),
    }
    observed_schema = metadata.get("latent_feature_schema", metadata.get("feature_schema"))
    observed = {
        "representation_kind": metadata.get("representation_kind"),
        "model_fingerprint": metadata.get("model_fingerprint"),
        "encoder_fingerprint": metadata.get("encoder_fingerprint"),
        "model_version": metadata.get("model_version"),
        "latent_feature_schema": tuple(observed_schema) if isinstance(observed_schema, (list, tuple)) else observed_schema,
    }
    mismatches = [name for name, value in expected.items() if observed.get(name) != value]
    if not isinstance(metadata.get("dataset_fingerprint"), str) or not metadata["dataset_fingerprint"].strip():
        mismatches.append("dataset_fingerprint")
    current_dataset = state.provenance.get("dataset_fingerprint")
    if current_dataset and metadata.get("dataset_fingerprint") != current_dataset:
        mismatches.append("dataset_fingerprint")
    if mismatches:
        raise ValueError(f"historical multimodal latent provenance is incompatible: {', '.join(mismatches)}")


def _latent_metadata_from_rows(observations: pd.DataFrame) -> Mapping[str, Any] | None:
    names = (
        "representation_kind", "model_fingerprint", "encoder_fingerprint", "model_version",
        "latent_feature_schema", "dataset_fingerprint",
    )
    if any(name not in observations for name in names):
        return None
    values: dict[str, Any] = {}
    for name in names:
        column = observations[name].tolist()
        if not column or any(item != column[0] for item in column[1:]):
            return None
        values[name] = column[0]
    return values

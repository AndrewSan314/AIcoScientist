from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..stages import ProcessStage


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (Enum, bool, str, Mapping, list, tuple)):
        return False
    if isinstance(value, Real):
        return math.isnan(float(value))
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _canonical_value(value: Any) -> Any:
    if _is_missing_scalar(value):
        return {"__missing__": True}
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("identity values must be finite")
        return numeric
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported identity value type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_control_action_id(controls: Mapping[str, Any]) -> str:
    """Deterministic identity for a controllable action, independent of source row."""
    return hashlib.sha256(_canonical_json(dict(controls)).encode("utf-8")).hexdigest()


def context_provenance_fingerprint(
    feature_values: Mapping[str, float], decision_stage: ProcessStage, representation_kind: str, *,
    semantic_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash semantic state identity, excluding runtime-only fields such as timestamps."""
    semantic = _semantic_metadata(semantic_metadata or {})
    semantic.setdefault("source_stage_ids", ())
    payload = {
        "decision_stage": decision_stage.value, "representation_kind": representation_kind,
        "features": dict(feature_values), "semantic_metadata": semantic,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _semantic_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_metadata(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in ("timestamp", "created_at", "updated_at", "retrieved_at", "started_at", "ended_at", "wall_time"))
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_metadata(item) for item in value]
    return value


def contextual_candidate_instance_id(context_fingerprint: str, control_action_id: str, stage: ProcessStage) -> str:
    payload = {"context_provenance_fingerprint": context_fingerprint, "control_action_id": control_action_id, "stage": stage.value}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class ModelValidationStatus(str, Enum):
    TEST_ONLY = "TEST_ONLY"
    TRAINED_UNVALIDATED = "TRAINED_UNVALIDATED"
    SOURCE_BACKED_VALIDATED = "SOURCE_BACKED_VALIDATED"


@dataclass(frozen=True)
class OptimizationState:
    """Numeric, provenance-carrying state consumed by contextual optimization."""

    feature_names: tuple[str, ...]
    feature_values: Mapping[str, float]
    decision_stage: ProcessStage
    source_stage_ids: tuple[str, ...]
    provenance_fingerprint: str
    representation_kind: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation_status: ModelValidationStatus = ModelValidationStatus.TEST_ONLY

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("optimization state needs unique numeric feature names")
        if set(self.feature_values) != set(self.feature_names):
            raise ValueError("optimization state feature names and values must match exactly")
        if not isinstance(self.decision_stage, ProcessStage) or not self.provenance_fingerprint.strip():
            raise ValueError("optimization state needs a decision stage and provenance fingerprint")
        if self.representation_kind not in {"scalar_horizon", "multimodal_stage_state", "scalar_plus_multimodal_stage_state", "multimodal_fused_baseline"}:
            raise ValueError(f"unsupported optimization state representation: {self.representation_kind!r}")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in self.feature_values.values()):
            raise ValueError("optimization state features must be finite numeric values")
        try:
            status = ModelValidationStatus(self.validation_status)
        except ValueError as exc:
            raise ValueError(f"unsupported optimization state validation status: {self.validation_status!r}") from exc
        object.__setattr__(self, "validation_status", status)

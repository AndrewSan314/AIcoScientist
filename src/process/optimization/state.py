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


def _canonical_value(value: Any) -> Any:
    if value is None or value is np.nan or value is pd.NA:
        return None
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


def context_provenance_fingerprint(feature_values: Mapping[str, float], decision_stage: ProcessStage, representation_kind: str) -> str:
    payload = {"decision_stage": decision_stage.value, "representation_kind": representation_kind, "features": dict(feature_values)}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def contextual_candidate_instance_id(context_fingerprint: str, control_action_id: str, stage: ProcessStage) -> str:
    payload = {"context_provenance_fingerprint": context_fingerprint, "control_action_id": control_action_id, "stage": stage.value}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("optimization state needs unique numeric feature names")
        if set(self.feature_values) != set(self.feature_names):
            raise ValueError("optimization state feature names and values must match exactly")
        if not isinstance(self.decision_stage, ProcessStage) or not self.provenance_fingerprint.strip():
            raise ValueError("optimization state needs a decision stage and provenance fingerprint")
        if self.representation_kind not in {"scalar_horizon", "multimodal_latent", "scalar_plus_multimodal"}:
            raise ValueError(f"unsupported optimization state representation: {self.representation_kind!r}")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in self.feature_values.values()):
            raise ValueError("optimization state features must be finite numeric values")

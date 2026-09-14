"""Source-backed manufacturing-success classifiers for constrained proposals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


_FAILURE_LABELS = frozenset({"mixing_failure", "coating_defect", "out_of_spec", "adhesion_failure", "assembly_rejection", "downstream_test_missing_due_to_process_failure"})


@dataclass(frozen=True)
class PredictionProbability:
    probability: float
    calibration_status: str
    source_model: str
    sample_count: int
    supported_stage: str | None


class ManufacturabilityModel:
    """A binary learned feasibility model; never infer probability from constraints."""

    def __init__(self, *, family: Literal["extra_trees", "random_forest"] = "extra_trees", label_name: str = "manufacturing_success", supported_stage: str | None = None, seed: int = 42) -> None:
        if family not in {"extra_trees", "random_forest"} or not label_name.strip():
            raise ValueError("manufacturability family and label name are required")
        self.family, self.label_name, self.supported_stage, self.seed = family, label_name, supported_stage, seed

    def fit(self, features: pd.DataFrame, labels: Sequence[bool | str]) -> "ManufacturabilityModel":
        if len(features) != len(labels) or features.empty:
            raise ValueError("features and manufacturability labels must be non-empty and aligned")
        numeric = features.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("manufacturability features must be finite numeric values")
        values = np.asarray([self._label(value) for value in labels], dtype=int)
        if set(values) != {0, 1}:
            raise ValueError("manufacturability training requires both success and failure labels")
        classifier = ExtraTreesClassifier(n_estimators=200, random_state=self.seed, n_jobs=1) if self.family == "extra_trees" else RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=1)
        self.feature_names, self.sample_count, self.classifier = tuple(features.columns), len(features), classifier.fit(numeric, values)
        payload = {"family": self.family, "label_name": self.label_name, "supported_stage": self.supported_stage, "seed": self.seed, "sample_count": self.sample_count, "feature_names": self.feature_names}
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self

    @staticmethod
    def _label(value: bool | str) -> int:
        if value is True or value == "manufacturing_success":
            return 1
        if value is False or value in _FAILURE_LABELS:
            return 0
        raise ValueError(f"unsupported manufacturability label: {value!r}")

    def predict(self, candidates: pd.DataFrame) -> list[PredictionProbability]:
        if not hasattr(self, "classifier"):
            raise RuntimeError("manufacturability model must be fitted before prediction")
        missing = [name for name in self.feature_names if name not in candidates]
        if missing:
            raise ValueError(f"candidate pool lacks manufacturability features: {missing}")
        X = candidates.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError("manufacturability candidate features must be finite numeric values")
        index = list(self.classifier.classes_).index(1)
        return [PredictionProbability(float(value), "TREE_ENSEMBLE_UNCALIBRATED", type(self.classifier).__name__, self.sample_count, self.supported_stage) for value in self.classifier.predict_proba(X)[:, index]]

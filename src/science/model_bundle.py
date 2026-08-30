from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.science.direct_baseline import DirectPerformanceModel
from src.science.provenance import ScientificModelProvenance
from src.science.two_stage import TwoStageScientificModel


@dataclass
class ScientificModelBundle:
    """Encapsulates all trained direct, Stage-A, and Stage-B models, dataset specs, provenance, and evaluations."""

    direct_model: DirectPerformanceModel
    two_stage_model: TwoStageScientificModel
    spec: DatasetSpec
    two_stage_spec: TwoStageModelSpec
    provenance: ScientificModelProvenance
    evaluation_report: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def save(self, path: Path | str) -> None:
        """Serializes the complete scientific model bundle to disk using joblib."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)

    @classmethod
    def load(cls, path: Path | str) -> ScientificModelBundle:
        """Loads and validates a serialized scientific model bundle."""
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"Loaded object is of type {type(bundle)!r}, expected {cls.__name__}")
        return bundle

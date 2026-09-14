from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .core import ProcessSurrogateSample


@dataclass(frozen=True)
class MultiFidelityFitReport:
    status: str
    target: str
    low_fidelity: str
    high_fidelity: str
    paired_sample_count: int
    unmatched_sample_count: int
    model_fingerprint: str


class PairedFidelityResidualCorrection:
    """Low-to-high affine correction fitted only on exact source recipe pairs."""

    def __init__(self, low_fidelity: str, high_fidelity: str, *, min_pairs: int = 3) -> None:
        if not low_fidelity.strip() or not high_fidelity.strip() or low_fidelity == high_fidelity or min_pairs < 3:
            raise ValueError("distinct fidelity names and at least three pairs are required")
        self.low_fidelity, self.high_fidelity, self.min_pairs = low_fidelity, high_fidelity, min_pairs

    def fit(self, samples: Sequence[ProcessSurrogateSample], *, target: str) -> "PairedFidelityResidualCorrection":
        grouped: dict[tuple[str, str | None, str, str], dict[str, ProcessSurrogateSample]] = {}
        for sample in samples:
            if sample.fidelity not in {self.low_fidelity, self.high_fidelity} or target not in sample.targets:
                continue
            if sample.recipe_id is None:
                continue
            key = (sample.source_dataset, sample.dataset_manifest_fingerprint, sample.recipe_id, sample.stage.value)
            pair = grouped.setdefault(key, {})
            if sample.fidelity in pair:
                raise ValueError(f"ambiguous {sample.fidelity!r} fidelity pair for recipe {sample.recipe_id!r}")
            pair[sample.fidelity] = sample
        pairs = [pair for pair in grouped.values() if set(pair) == {self.low_fidelity, self.high_fidelity}]
        unmatched = sum(len(pair) for pair in grouped.values() if set(pair) != {self.low_fidelity, self.high_fidelity})
        if len(pairs) < self.min_pairs:
            raise ValueError("MULTIFIDELITY_CORRECTION_NOT_EVALUATABLE: insufficient compatible source recipe pairs")
        for pair in pairs:
            low, high = pair[self.low_fidelity], pair[self.high_fidelity]
            if low.controls != high.controls:
                raise ValueError(f"paired recipe {low.recipe_id!r} has incompatible controls")
        low_y = np.asarray([pair[self.low_fidelity].targets[target] for pair in pairs], dtype=float)
        high_y = np.asarray([pair[self.high_fidelity].targets[target] for pair in pairs], dtype=float)
        design = np.column_stack((low_y, np.ones(len(low_y))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, high_y, rcond=None)
        if rank < 2:
            coefficients = np.asarray([0.0, high_y.mean()])
        residuals = high_y - design @ coefficients
        self.slope, self.intercept = map(float, coefficients)
        self.residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        pair_ids = tuple(sorted((pair[self.low_fidelity].sample_id, pair[self.high_fidelity].sample_id) for pair in pairs))
        payload = {"target": target, "low_fidelity": self.low_fidelity, "high_fidelity": self.high_fidelity, "slope": self.slope, "intercept": self.intercept, "residual_std": self.residual_std, "pairs": pair_ids}
        self.report = MultiFidelityFitReport("EVALUATED_PAIRED_SOURCE", target, self.low_fidelity, self.high_fidelity, len(pairs), unmatched, hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        return self

    def predict(self, X: np.ndarray, *, target: str) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "report"):
            raise ValueError("fit paired fidelity correction before prediction")
        if target != self.report.target:
            raise ValueError("correction target differs from the fitted target")
        values = np.asarray(X, dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("low-fidelity predictions must be a finite vector or one-column matrix")
        return self.slope * values + self.intercept, np.full(len(values), self.residual_std)

    @property
    def artifact_metadata(self) -> dict[str, object]:
        if not hasattr(self, "report"):
            raise ValueError("fit paired fidelity correction before reading metadata")
        return {"fidelity_semantics": "paired_low_to_high_affine_residual_correction", **self.report.__dict__}

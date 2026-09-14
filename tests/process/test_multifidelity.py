from __future__ import annotations

import numpy as np
import pytest

from src.process.stages import ProcessStage
from src.process.surrogates import PairedFidelityResidualCorrection, ProcessSurrogateSample


def _sample(recipe: str, fidelity: str, value: float, *, controls: dict[str, float] | None = None) -> ProcessSurrogateSample:
    return ProcessSurrogateSample(f"{fidelity}-{recipe}", f"run-{recipe}", recipe, "synthetic-test", "SYNTHETIC_TEST_ONLY", ProcessStage.COATING, recipe, controls or {"speed": 1.0}, fidelity=fidelity, targets={"capacity": value}, dataset_manifest_fingerprint="fixture")


def test_paired_fidelity_correction_uses_only_exact_recipe_pairs() -> None:
    samples = [item for recipe, low, high in (("a", 1.0, 3.0), ("b", 2.0, 5.0), ("c", 3.0, 7.0)) for item in (_sample(recipe, "LOW", low), _sample(recipe, "HIGH", high))]
    model = PairedFidelityResidualCorrection("LOW", "HIGH").fit(samples, target="capacity")
    mean, std = model.predict(np.array([4.0]), target="capacity")
    assert mean.tolist() == pytest.approx([9.0])
    assert std.shape == (1,)
    assert model.artifact_metadata["paired_sample_count"] == 3


def test_paired_fidelity_correction_rejects_unmatched_or_incompatible_pairs() -> None:
    with pytest.raises(ValueError, match="NOT_EVALUATABLE"):
        PairedFidelityResidualCorrection("LOW", "HIGH").fit([_sample("a", "LOW", 1.0)], target="capacity")
    samples = [item for recipe, low, high in (("a", 1.0, 3.0), ("b", 2.0, 5.0), ("c", 3.0, 7.0)) for item in (_sample(recipe, "LOW", low), _sample(recipe, "HIGH", high, controls={"speed": 2.0} if recipe == "c" else None))]
    with pytest.raises(ValueError, match="incompatible controls"):
        PairedFidelityResidualCorrection("LOW", "HIGH").fit(samples, target="capacity")

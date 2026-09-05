from __future__ import annotations

from typing import Any

import numpy as np


class MedianThresholdSegmentationBaseline:
    """Dependency-light median-threshold segmentation; no CAMEO code is used."""

    name = "median_threshold_segmentation"
    version = "1.0.0"

    def segment(self, image: Any) -> dict[str, Any]:
        values = np.asarray(image, dtype=np.float64)
        if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("segmentation input must be a finite two-dimensional image")
        threshold = float(np.median(values))
        mask = values > threshold
        return {
            "mask": mask,
            "threshold": threshold,
            "foreground_fraction": float(np.mean(mask)),
            "reference_only": True,
            "limitation": "unvalidated classical threshold baseline; not a trained segmentation model",
        }


__all__ = ["MedianThresholdSegmentationBaseline"]

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

DEFAULT_NUM_PCA_COMPONENTS = 8
DEFAULT_GRID_POINTS = 450


class XRDRepresentationExtractor:
    """Extracts standardized, low-dimensional structural embeddings from real XRD diffractograms.

    LEAKAGE CONTRACT:
    - PCA is fitted strictly on revealed XRD spectra available at the current campaign step.
    - When N_revealed < min_pca_samples (default 3), uses a deterministic coarse-binned
      intensity representation without fitting or leaking unobserved spectra.
    - Guaranteed to be deterministic and reproducible.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_NUM_PCA_COMPONENTS,
        min_pca_samples: int = 3,
        num_grid_points: int = DEFAULT_GRID_POINTS,
    ) -> None:
        self.n_components = n_components
        self.min_pca_samples = min_pca_samples
        self.num_grid_points = num_grid_points
        self._pca: PCA | None = None
        self._fitted_sample_count: int = 0

    @property
    def is_pca_fitted(self) -> bool:
        return self._pca is not None

    def fit(self, revealed_spectra: Sequence[np.ndarray | list[float]]) -> XRDRepresentationExtractor:
        """Fits PCA dimensionality reduction on revealed standardized spectra."""
        if len(revealed_spectra) < self.min_pca_samples:
            self._pca = None
            self._fitted_sample_count = len(revealed_spectra)
            return self

        X = np.asarray(revealed_spectra, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array of spectra (n_samples, n_points), got shape {X.shape}")

        effective_components = min(self.n_components, len(revealed_spectra) - 1, X.shape[1])
        if effective_components < 1:
            effective_components = 1

        self._pca = PCA(n_components=effective_components, random_state=42)
        self._pca.fit(X)
        self._fitted_sample_count = len(revealed_spectra)
        return self

    def transform(self, spectrum: np.ndarray | list[float]) -> np.ndarray:
        """Transforms a single standardized spectrum into a feature embedding."""
        spec = np.asarray(spectrum, dtype=np.float64).flatten()

        if self._pca is not None:
            emb = self._pca.transform(spec.reshape(1, -1))[0]
            # Pad to fixed n_components if effective_components < n_components
            if len(emb) < self.n_components:
                padded = np.zeros(self.n_components, dtype=np.float64)
                padded[: len(emb)] = emb
                return padded
            return emb
        else:
            # Deterministic coarse binning fallback (e.g. 8 coarse region means)
            # Divides spectrum into n_components equal bins
            bin_size = max(1, len(spec) // self.n_components)
            bins = []
            for i in range(self.n_components):
                start = i * bin_size
                end = (i + 1) * bin_size if i < self.n_components - 1 else len(spec)
                if start < len(spec):
                    bins.append(float(np.mean(spec[start:end])))
                else:
                    bins.append(0.0)
            return np.asarray(bins, dtype=np.float64)

    def transform_batch(
        self,
        spectra_map: Mapping[str, np.ndarray | list[float]],
    ) -> dict[str, np.ndarray]:
        """Transforms a mapping of candidate_id -> spectrum into candidate_id -> embedding."""
        return {cid: self.transform(spec) for cid, spec in spectra_map.items()}

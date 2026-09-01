from __future__ import annotations

import copy
import hashlib
import logging
from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA

from src.science.representation import RepresentationSnapshot

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
    - Supports representation snapshots and frozen-basis transformations during Bayesian evidence updates.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_NUM_PCA_COMPONENTS,
        min_pca_samples: int = 3,
        num_grid_points: int = DEFAULT_GRID_POINTS,
        representation_id: str = "xrd_pca",
    ) -> None:
        self.n_components = n_components
        self.min_pca_samples = min_pca_samples
        self.num_grid_points = num_grid_points
        self.representation_id = representation_id
        self._pca: PCA | None = None
        self._fitted_sample_count: int = 0
        self._version: int = 0

    @property
    def is_pca_fitted(self) -> bool:
        return self._pca is not None

    @property
    def version(self) -> int:
        return self._version

    def get_fingerprint(self) -> str:
        """Computes a deterministic cryptographic fingerprint of the current representation basis."""
        if self._pca is not None and hasattr(self._pca, "components_"):
            hasher = hashlib.sha256()
            hasher.update(f"pca_dim_{self.n_components}_samples_{self._fitted_sample_count}_".encode())
            hasher.update(self._pca.components_.tobytes())
            hasher.update(self._pca.mean_.tobytes())
            return hasher.hexdigest()
        else:
            return f"coarse_binning_{self.n_components}_samples_{self._fitted_sample_count}"

    def get_snapshot(self, representation_id: str | None = None, modality_name: str = "XRD") -> RepresentationSnapshot:
        """Returns an immutable snapshot representing the frozen coordinate basis at this moment."""
        rep_id = representation_id if representation_id is not None else self.representation_id
        frozen_pca = copy.deepcopy(self._pca) if self._pca is not None else None
        return RepresentationSnapshot(
            representation_id=rep_id,
            version=self._version,
            modality_name=modality_name,
            fingerprint=self.get_fingerprint(),
            metadata={
                "fitted_sample_count": self._fitted_sample_count,
                "is_pca_fitted": self.is_pca_fitted,
                "n_components": self.n_components,
                "_frozen_pca": frozen_pca,
            },
        )

    def fit(self, revealed_spectra: Sequence[np.ndarray | list[float]]) -> XRDRepresentationExtractor:
        """Fits PCA dimensionality reduction on revealed standardized spectra."""
        if len(revealed_spectra) < self.min_pca_samples:
            self._pca = None
            self._fitted_sample_count = len(revealed_spectra)
            self._version += 1
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
        self._version += 1
        return self

    def transform(self, spectrum: np.ndarray | list[float]) -> np.ndarray:
        """Transforms a single standardized spectrum into a feature embedding."""
        return self._transform_internal(spectrum, pca_model=self._pca)

    def transform_with_snapshot(
        self,
        spectrum: np.ndarray | list[float],
        snapshot: RepresentationSnapshot,
    ) -> np.ndarray:
        """Transforms a spectrum using the frozen PCA basis embedded in the representation snapshot."""
        frozen_pca = snapshot.metadata.get("_frozen_pca")
        return self._transform_internal(spectrum, pca_model=frozen_pca)

    def _transform_internal(
        self,
        spectrum: np.ndarray | list[float],
        pca_model: PCA | None,
    ) -> np.ndarray:
        spec = np.asarray(spectrum, dtype=np.float64).flatten()

        if pca_model is not None and hasattr(pca_model, "components_"):
            emb = pca_model.transform(spec.reshape(1, -1))[0]
            if len(emb) < self.n_components:
                padded = np.zeros(self.n_components, dtype=np.float64)
                padded[: len(emb)] = emb
                return padded
            return emb
        else:
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

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class RepresentationMismatchError(ValueError):
    """Raised when an evidence update is attempted across incompatible representation coordinate bases."""

    pass


@dataclass(frozen=True)
class RepresentationSnapshot:
    """Immutable snapshot of a modality representation basis.

    Attributes
    ----------
    representation_id : str
        Identifier of the representation model/basis (e.g. 'auirh_xrd_pca', 'alab_xrd_pca').
    version : int
        Monotonically increasing version index of the representation basis.
    modality_name : str
        Name of the measurement modality this representation applies to (e.g. 'XRD').
    fingerprint : str
        Deterministic cryptographic hash or unique identifier of the exact representation basis state.
    metadata : dict[str, Any]
        Additional diagnostic metadata (e.g. fitted sample count, PCA explained variance).
    """

    representation_id: str
    version: int
    modality_name: str
    fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObservationRepresentationManager(Protocol):
    """Protocol for domain-level measurement representation lifecycle managers."""

    def get_representation_snapshot(self, modality_name: str) -> RepresentationSnapshot | None:
        """Returns an immutable snapshot of the current representation basis for a modality."""
        ...

    def transform_with_snapshot(
        self,
        modality_name: str,
        raw_observation: Any,
        snapshot: RepresentationSnapshot,
    ) -> Any:
        """Transforms a raw measurement using the specific frozen representation snapshot basis."""
        ...

    def update_representation_after_evidence(
        self,
        modality_name: str,
        candidate_id: str,
        raw_observation: Any,
    ) -> None:
        """Updates the representation basis using newly confirmed evidence (called strictly after Bayesian update)."""
        ...

    def get_current_observations(self) -> Mapping[str, Mapping[str, Any]]:
        """Returns all currently revealed candidate observations in the latest representation basis."""
        ...

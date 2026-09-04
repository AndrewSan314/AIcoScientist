from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def raw_artifact_sha256(raw_artifact: bytes | bytearray | str | Path) -> str:
    """Hashes raw bytes or a referenced file without changing the artifact."""
    if isinstance(raw_artifact, (str, Path)):
        with Path(raw_artifact).open("rb") as handle:
            payload = handle.read()
    else:
        payload = bytes(raw_artifact)
    return hashlib.sha256(payload).hexdigest()


def build_observable_provenance(
    raw_artifact_ref: str | None,
    extractor_name: str,
    extractor_version: str,
    *,
    raw_artifact_hash: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    model_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Builds the minimum provenance envelope needed to reproduce an observable."""
    result: dict[str, Any] = {
        "raw_artifact_ref": raw_artifact_ref,
        "raw_artifact_sha256": raw_artifact_hash,
        "extractor_name": extractor_name,
        "extractor_version": extractor_version,
        "configuration": dict(configuration or {}),
        "model_checkpoint": model_checkpoint,
    }
    return result


__all__ = ["build_observable_provenance", "raw_artifact_sha256"]

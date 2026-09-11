from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PINNED_COMMIT, PINNED_SOURCE_TREE_HASH, SOURCE_URL, ArtisticRunConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(source_root: Path, *, files: list[Path] | None = None) -> dict[str, Any]:
    ArtisticRunConfig(source_root=source_root).verify_source_pin()
    def git(*args: str) -> str:
        completed = subprocess.run(["git", "-C", str(source_root), *args], capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    selected = files or []
    return {
        "source_url": SOURCE_URL, "pinned_upstream_commit": PINNED_COMMIT,
        "checked_out_commit": git("rev-parse", "HEAD"), "source_tree_hash": git("rev-parse", "HEAD:NMC/Updated version"),
        "expected_source_tree_hash": PINNED_SOURCE_TREE_HASH,
        "source_repository_tree_hash": git("rev-parse", "HEAD^{tree}"), "retrieved_at": datetime.now(UTC).isoformat(),
        "source_file_hashes": {str(path.relative_to(source_root)).replace("\\", "/"): sha256(path) for path in selected if path.is_file()},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temp:
        temp.write(data)
        temp.flush()
        os.fsync(temp.fileno())
        temporary = temp.name
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

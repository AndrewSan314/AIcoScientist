from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PINNED_COMMIT, SOURCE_URL


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance(source_root: Path, *, files: list[Path] | None = None) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(["git", "-C", str(source_root), *args], capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    selected = files or []
    return {
        "source_url": SOURCE_URL, "pinned_upstream_commit": PINNED_COMMIT,
        "checked_out_commit": git("rev-parse", "HEAD"), "source_tree_hash": git("rev-parse", "HEAD:NMC/Updated version"),
        "source_repository_tree_hash": git("rev-parse", "HEAD^{tree}"), "retrieved_at": datetime.now(UTC).isoformat(),
        "source_file_hashes": {str(path.relative_to(source_root)).replace("\\", "/"): sha256(path) for path in selected if path.is_file()},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

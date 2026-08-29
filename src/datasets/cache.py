from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            sha.update(chunk)
    return sha.hexdigest()


def write_processed_manifest(
    processed_dir: Path,
    dataset: str,
    source_version: str,
    raw_manifest_path: Path | None,
    adapter_schema_version: str,
    feature_horizon: int | None,
    processed_files: list[Path],
) -> dict[str, Any]:
    """Writes processed_manifest.json recording cache validity and provenance."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_hash = compute_file_sha256(raw_manifest_path) if raw_manifest_path and raw_manifest_path.exists() else None

    files_dict: dict[str, str] = {}
    for pfile in processed_files:
        if pfile.exists():
            rel_name = pfile.relative_to(processed_dir).as_posix() if pfile.is_relative_to(processed_dir) else pfile.name
            files_dict[rel_name] = compute_file_sha256(pfile)

    manifest_data = {
        "dataset": dataset,
        "source_version": source_version,
        "raw_manifest_hash": raw_hash,
        "adapter_schema_version": adapter_schema_version,
        "feature_horizon": feature_horizon,
        "processed_files": files_dict,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    manifest_path = processed_dir / "processed_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def validate_processed_cache(
    processed_dir: Path,
    raw_manifest_path: Path | None,
    adapter_schema_version: str,
    feature_horizon: int | None,
    expected_files: list[str],
) -> bool:
    """Validates processed cache metadata without rehashing large raw files."""
    manifest_path = processed_dir / "processed_manifest.json"
    if not manifest_path.exists():
        return False

    # Check expected files exist on disk
    for fname in expected_files:
        fpath = processed_dir / fname
        if not fpath.exists():
            return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception:
        return False

    # Check adapter schema version
    if manifest_data.get("adapter_schema_version") != adapter_schema_version:
        return False

    # Check feature horizon
    if manifest_data.get("feature_horizon") != feature_horizon:
        return False

    # Check raw manifest hash
    if raw_manifest_path and raw_manifest_path.exists():
        current_raw_hash = compute_file_sha256(raw_manifest_path)
        if manifest_data.get("raw_manifest_hash") != current_raw_hash:
            return False
    elif manifest_data.get("raw_manifest_hash") is not None:
        return False

    # Check processed file hashes
    stored_files = manifest_data.get("processed_files", {})
    for fname in expected_files:
        if fname not in stored_files:
            return False
        fpath = processed_dir / fname
        if compute_file_sha256(fpath) != stored_files[fname]:
            return False

    return True

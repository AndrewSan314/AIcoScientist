from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.cache import validate_processed_cache, write_processed_manifest


def test_processed_manifest_write_and_validate(tmp_path: Path):
    raw_manifest = tmp_path / "raw_manifest.json"
    raw_manifest.write_text(json.dumps({"dataset": "test", "files": []}), encoding="utf-8")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    f1 = processed_dir / "table1.csv"
    f1.write_text("a,b\n1,2\n", encoding="utf-8")

    write_processed_manifest(
        processed_dir=processed_dir,
        dataset="test",
        source_version="v1",
        raw_manifest_path=raw_manifest,
        adapter_schema_version="1.0.0",
        feature_horizon=100,
        processed_files=[f1],
    )

    # Valid check
    assert validate_processed_cache(
        processed_dir=processed_dir,
        raw_manifest_path=raw_manifest,
        adapter_schema_version="1.0.0",
        feature_horizon=100,
        expected_files=["table1.csv"],
    ) is True

    # Invalidate on schema version change
    assert validate_processed_cache(
        processed_dir=processed_dir,
        raw_manifest_path=raw_manifest,
        adapter_schema_version="2.0.0",
        feature_horizon=100,
        expected_files=["table1.csv"],
    ) is False

    # Invalidate on horizon change
    assert validate_processed_cache(
        processed_dir=processed_dir,
        raw_manifest_path=raw_manifest,
        adapter_schema_version="1.0.0",
        feature_horizon=200,
        expected_files=["table1.csv"],
    ) is False

    # Invalidate on raw manifest change (tampering with raw_manifest)
    raw_manifest.write_text(json.dumps({"dataset": "test", "files": ["new_raw_file"]}), encoding="utf-8")
    assert validate_processed_cache(
        processed_dir=processed_dir,
        raw_manifest_path=raw_manifest,
        adapter_schema_version="1.0.0",
        feature_horizon=100,
        expected_files=["table1.csv"],
    ) is False

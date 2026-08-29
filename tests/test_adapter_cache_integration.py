from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.datasets.dynamic_cycling import DynamicCyclingAdapter, DYNAMIC_CYCLING_FEATURE_COLUMNS


def _create_synthetic_raw_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = pd.DataFrame({
        "cell_name": ["cell_001", "cell_002", "cell_003", "cell_004"],
        "protocol_name": ["P01", "P01", "P02", "P02"],
        "protocol_type": ["Fast", "Fast", "Normal", "Normal"],
        "protocol_variant": [1, 1, 2, 2],
        "avg_crate_exp": [2.5, 2.5, 1.0, 1.0],
    })
    metadata.to_pickle(raw_dir / "metadata.pkl")

    proto_dict = {
        "Average Current": [2.5, 2.5, 1.0, 1.0],
        "Normalized Current Variance": [0.1, 0.1, 0.05, 0.05],
        "Maximum Discharge Current": [5.0, 5.0, 2.0, 2.0],
        "Relative Charge Fraction": [0.8, 0.8, 0.8, 0.8],
        "Rest Fraction at High SOC": [0.2, 0.2, 0.2, 0.2],
        "Rest SOC": [0.9, 0.9, 0.9, 0.9],
        "Peak Frequency 1": [0.01, 0.01, 0.005, 0.005],
        "Peak Frequency 2": [0.02, 0.02, 0.01, 0.01],
    }
    proto_df = pd.DataFrame(proto_dict, index=["cell_001", "cell_002", "cell_003", "cell_004"])
    proto_df.to_pickle(raw_dir / "protocol_features.pkl")

    soh90 = pd.DataFrame({
        "cell_name": ["cell_001", "cell_002", "cell_003", "cell_004"],
        "EFCs (with Diagnostic)": [500.0, 520.0, 800.0, 810.0],
        "Cycles": [450, 460, 750, 760],
    })
    soh90.to_csv(raw_dir / "soh90.csv", index=False)


def test_dynamic_adapter_load_cells_and_protocols_cache_invalidation(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_manifest_path = tmp_path / "manifest.json"

    _create_synthetic_raw_dataset(raw_dir)
    raw_manifest_path.write_text(json.dumps({"dataset": "dynamic_cycling_2024", "files": []}), encoding="utf-8")

    adapter = DynamicCyclingAdapter(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_manifest_path=raw_manifest_path,
        expected_records=4,
        expected_protocols=2,
    )

    # 1. Initial load generates cache (4 cells, 2 protocols)
    cells_df = adapter.load_cells(force_recompute=True)
    assert len(cells_df) == 4
    protocols_df = adapter.load_protocols()
    assert len(protocols_df) == 2

    manifest_file = processed_dir / "processed_manifest.json"
    assert manifest_file.exists()

    # 2. Plant stale data in cells.csv without updating manifest
    stale_cells = cells_df.copy()
    stale_cells["efc_lifetime"] = 9999.0
    stale_cells.to_csv(processed_dir / "cells.csv", index=False)

    # Calling load_cells must detect hash mismatch on cells.csv and recompute from raw!
    fresh_cells = adapter.load_cells()
    assert fresh_cells["efc_lifetime"].max() < 1000.0  # Proves stale cache was rejected and recomputed!

    # 3. Modify raw manifest -> Calling load_protocols must detect raw manifest hash change and recompute
    raw_manifest_path.write_text(json.dumps({"dataset": "dynamic_cycling_2024", "files": ["tampered"]}), encoding="utf-8")
    
    # Plant stale protocols
    stale_protocols = protocols_df.copy()
    stale_protocols["target_mean"] = 8888.0
    stale_protocols.to_csv(processed_dir / "protocols.csv", index=False)

    fresh_protocols = adapter.load_protocols()
    assert fresh_protocols["target_mean"].max() < 1000.0  # Proves stale cache was rejected on raw manifest change!


def test_dynamic_adapter_schema_version_bump_invalidates_cache(tmp_path: Path):
    """Proves that a processed manifest built with adapter schema version 2.0.0 is rejected by 3.0.0."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_manifest_path = tmp_path / "manifest.json"

    _create_synthetic_raw_dataset(raw_dir)
    raw_manifest_path.write_text(json.dumps({"dataset": "dynamic_cycling_2024", "files": []}), encoding="utf-8")

    adapter = DynamicCyclingAdapter(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_manifest_path=raw_manifest_path,
        expected_records=4,
        expected_protocols=2,
    )
    # Generate initial cache
    _ = adapter.load_cells(force_recompute=True)
    manifest_file = processed_dir / "processed_manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    from src.datasets.dynamic_cycling import ADAPTER_SCHEMA_VERSION
    assert manifest_data["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION

    # Plant version 2.0.0 in manifest and plant stale data
    manifest_data["adapter_schema_version"] = "2.0.0"
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    stale_cells = pd.read_csv(processed_dir / "cells.csv")
    stale_cells["efc_lifetime"] = 7777.0
    stale_cells.to_csv(processed_dir / "cells.csv", index=False)

    # Re-computing hash in manifest for the stale cells to simulate a valid v2.0.0 manifest
    from src.datasets.cache import compute_file_sha256
    manifest_data["processed_files"]["cells.csv"] = compute_file_sha256(processed_dir / "cells.csv")
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Loading cells with adapter MUST reject the v2.0.0 manifest and recompute!
    fresh_cells = adapter.load_cells()
    assert fresh_cells["efc_lifetime"].max() < 1000.0

    # Verify that the manifest was updated to ADAPTER_SCHEMA_VERSION
    updated_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert updated_manifest["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION

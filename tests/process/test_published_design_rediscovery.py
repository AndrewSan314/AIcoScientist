"""Tests auditing published design status and Mode 2 unrecoverability."""

from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_drakopoulos_source_audit_records_mode_2_status() -> None:
    audit_file = Path("outputs/rediscovery_audit/drakopoulos_source_audit.json")
    assert audit_file.is_file(), "outputs/rediscovery_audit/drakopoulos_source_audit.json missing"

    with open(audit_file) as f:
        data = json.load(f)

    assert data["dataset_id"] == "drakopoulos_graphite"
    assert data["candidate_pool_size"] == 13
    assert data["retrospective_best_recipe_id"] == "AS-104"
    assert pytest.approx(data["retrospective_best_recipe_capacity_mah"], abs=1e-4) == 11.04757

    mode_2 = data["mode_2_support"]
    assert mode_2["status"] == "PUBLISHED_OPTIMIZED_DESIGN_NOT_SOURCE_RECOVERABLE"
    assert "alchemite_model_training.txt" in mode_2["files_audited"]
    assert len(mode_2["description"]) > 20


def test_alchemite_script_is_training_invocation_only() -> None:
    raw_script = Path("data/external/drakopoulos_graphite/1/raw/alchemite_model_training.txt")
    if not raw_script.is_file():
        pytest.skip("Raw alchemite script not present in local path")

    content = raw_script.read_text()
    assert "api_models.models_id_train_put" in content
    assert "await_trained" in content
    # Script does not output an isolated recipe or optimization recommendation
    assert "optimal_recipe" not in content.lower()
    assert "recommended_cell" not in content.lower()


def test_source_observed_best_recipe_hierarchy() -> None:
    audit_file = Path("outputs/rediscovery_audit/drakopoulos_source_audit.json")
    with open(audit_file) as f:
        data = json.load(f)

    recipes = data["candidate_recipes"]
    assert len(recipes) == 13
    # Top 3 strictly ordered
    assert recipes[0]["run_id"] == "AS-104"
    assert recipes[1]["run_id"] == "AS-46"
    assert recipes[2]["run_id"] == "AS-21"
    assert recipes[0]["cell_capacity_mah"] > recipes[1]["cell_capacity_mah"] > recipes[2]["cell_capacity_mah"]

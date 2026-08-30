from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def get_git_provenance(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Extracts git commit, branch, and working-tree dirty status without crashing if git is absent."""
    repo_path = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    head_commit: str | None = None
    branch: str | None = None
    is_dirty = False
    diff_hash: str | None = None

    try:
        # 1. Get HEAD commit hash
        res_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if res_head.returncode == 0:
            head_commit = res_head.stdout.strip() or None

        # 2. Get current branch or tag
        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if res_branch.returncode == 0:
            branch = res_branch.stdout.strip() or None

        # 3. Check if working directory has modified tracked files
        res_status = subprocess.run(
            ["git", "status", "--porcelain", "-uno"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if res_status.returncode == 0 and res_status.stdout.strip():
            is_dirty = True
            # Compute deterministic sha256 of tracked diff
            res_diff = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if res_diff.returncode == 0 and res_diff.stdout:
                diff_hash = hashlib.sha256(res_diff.stdout.encode("utf-8")).hexdigest()
    except Exception:
        # Graceful degradation if git is unavailable
        pass

    return {
        "code_head_commit": head_commit,
        "branch": branch,
        "git_dirty": is_dirty,
        "git_diff_sha256": diff_hash,
    }


def get_environment_provenance() -> dict[str, Any]:
    """Captures runtime environment, OS, Python version, and core library versions."""
    import joblib
    import scipy
    import sklearn

    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "platform": platform.platform(),
    }


def compute_dataset_fingerprint(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    id_col: str | None = None,
) -> str:
    """Computes a deterministic SHA-256 fingerprint over the scientific training data content.

    Includes schema, column types, row ordering, non-oracle training values, and missingness.
    Preserves declared column order.
    """
    if df.empty:
        return hashlib.sha256(b"EMPTY_DATASET").hexdigest()[:16]

    cols_to_use = []
    if id_col and id_col in df.columns:
        cols_to_use.append(id_col)
    for c in feature_cols:
        if c in df.columns and c not in cols_to_use:
            cols_to_use.append(c)
    for c in target_cols:
        if c in df.columns and c not in cols_to_use:
            cols_to_use.append(c)

    sub_df = df[cols_to_use].copy()

    # Sort rows deterministically by id or existing columns
    if id_col and id_col in sub_df.columns:
        sub_df = sub_df.sort_values(by=id_col).reset_index(drop=True)
    else:
        sort_by = [c for c in cols_to_use if c in sub_df.columns]
        if sort_by:
            sub_df = sub_df.sort_values(by=sort_by).reset_index(drop=True)

    hasher = hashlib.sha256()
    hasher.update(json.dumps([(c, str(sub_df[c].dtype)) for c in cols_to_use]).encode("utf-8"))

    for _, row in sub_df.iterrows():
        row_repr = []
        for c in cols_to_use:
            val = row[c]
            if pd.isna(val):
                row_repr.append("NaN")
            elif isinstance(val, (float, np.floating)):
                f_val = float(val)
                if np.isnan(f_val):
                    row_repr.append("NaN")
                elif np.isposinf(f_val):
                    row_repr.append("+Inf")
                elif np.isneginf(f_val):
                    row_repr.append("-Inf")
                else:
                    # Deterministic IEEE 754 exact hex representation (no float rounding collapse)
                    row_repr.append(f_val.hex())
            elif isinstance(val, (int, np.integer)):
                row_repr.append(str(int(val)))
            else:
                row_repr.append(str(val).strip())
        hasher.update(("|".join(row_repr) + "\n").encode("utf-8"))

    return hasher.hexdigest()[:16]


def compute_search_space_fingerprint(search_space: Any) -> str:
    """Computes a deterministic SHA-256 fingerprint for a SearchSpace object.

    Handles ContinuousVariable, DiscreteVariable, DerivedVariable, and Constraints.
    Note: Python callable implementations are not serialized directly; supply an explicit
    provenance_id on custom derived variables or constraints for strict semantic tracking.
    """
    if search_space is None:
        return "NONE"
    items = []
    variables = getattr(search_space, "variables", [])
    for v in variables:
        v_type = getattr(v, "var_type", None) or type(v).__name__
        v_item: dict[str, Any] = {
            "name": getattr(v, "name", ""),
            "var_type": v_type,
        }
        if hasattr(v, "lower") and v.lower is not None:
            v_item["lower"] = float(v.lower).hex() if isinstance(v.lower, (int, float, np.number)) else str(v.lower)
        if hasattr(v, "upper") and v.upper is not None:
            v_item["upper"] = float(v.upper).hex() if isinstance(v.upper, (int, float, np.number)) else str(v.upper)
        if hasattr(v, "values") and v.values is not None:
            v_item["values"] = [
                float(val).hex() if isinstance(val, (int, float, np.number)) else str(val)
                for val in v.values
            ]
        if hasattr(v, "categories") and v.categories is not None:
            v_item["categories"] = [str(cat) for cat in v.categories]
        if hasattr(v, "provenance_id") and v.provenance_id is not None:
            v_item["provenance_id"] = str(v.provenance_id)
        items.append(v_item)

    derived_variables = getattr(search_space, "derived_variables", [])
    for d in derived_variables:
        d_type = getattr(d, "var_type", None) or type(d).__name__
        d_item: dict[str, Any] = {
            "name": getattr(d, "name", ""),
            "var_type": d_type,
            "depends_on": list(getattr(d, "depends_on", ())),  # Preserve declared depends_on order
        }
        if hasattr(d, "provenance_id") and d.provenance_id is not None:
            d_item["provenance_id"] = str(d.provenance_id)
        items.append(d_item)

    constraints = getattr(search_space, "constraints", [])
    for c in constraints:
        c_item: dict[str, Any] = {
            "name": getattr(c, "name", ""),
            "type": "constraint",
            "description": getattr(c, "description", ""),
        }
        if hasattr(c, "provenance_id") and c.provenance_id is not None:
            c_item["provenance_id"] = str(c.provenance_id)
        items.append(c_item)

    canon = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def compute_spec_fingerprint(
    spec: Any,
    two_stage_spec: Any | None = None,
    search_space: Any | None = None,
) -> str:
    """Computes a deterministic SHA-256 fingerprint over scientific dataset, two-stage specs, and search space.

    CRITICAL RULE: Declared feature ordering is semantic. Ordered lists (e.g. process_features,
    feature_columns) are NOT sorted to ensure [x1, x2] and [x2, x1] produce distinct fingerprints.
    """
    spec_dict: dict[str, Any] = {
        "name": getattr(spec, "name", ""),
        "id_column": getattr(spec, "id_column", ""),
        "candidate_id_column": getattr(spec, "candidate_id_column", ""),
        "entity_id_column": getattr(spec, "entity_id_column", None),
        "feature_columns": list(getattr(spec, "feature_columns", [])),
        "candidate_columns": list(getattr(spec, "candidate_columns", [])),
        "candidate_variables": list(getattr(spec, "candidate_variables", [])),
        "pre_experiment_features": list(getattr(spec, "pre_experiment_features", [])),
        "post_experiment_characterization": list(getattr(spec, "post_experiment_characterization", [])),
        "targets": list(getattr(spec, "targets", [])),
        "target_column": getattr(spec, "target_column", ""),
        "objective": getattr(spec, "objective", "maximize"),
        "oracle_columns": list(getattr(spec, "oracle_columns", [])),
        "constraints": [getattr(c, "to_dict", lambda: str(c))() for c in getattr(spec, "constraints", [])],
        "feature_horizon": getattr(spec, "feature_horizon", None),
        "source_dataset": getattr(spec, "source_dataset", None),
        "source_version": getattr(spec, "source_version", None),
    }
    if two_stage_spec is not None:
        spec_dict["two_stage"] = {
            "process_features": list(getattr(two_stage_spec, "process_features", [])),
            "characterization_targets": list(getattr(two_stage_spec, "characterization_targets", [])),
            "performance_targets": list(getattr(two_stage_spec, "performance_targets", [])),
        }
    if search_space is not None:
        spec_dict["search_space_fingerprint"] = compute_search_space_fingerprint(search_space)

    canon = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def build_benchmark_run_manifest(
    dataset_name: str,
    comparison_baseline_commit: str = "53a1c7241222105cdede343d5a155fdd5a97ee78",
    simulator_version: str | None = None,
    attia_source_commit: str | None = None,
    n_seeds: int | None = None,
    budgets: Any = None,
    strategies: list[str] | None = None,
    initial_policies: int | None = None,
    candidate_pool_size: int | None = None,
    duplicate_tolerance: float | None = None,
    n_jobs: int | None = None,
    reference_underestimated: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generates an explicit, non-ambiguous scientific run manifest for future benchmark executions."""
    git_info = get_git_provenance()
    env_info = get_environment_provenance()

    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "comparison_baseline_commit": comparison_baseline_commit,
        **git_info,
        **env_info,
        "simulator_version": simulator_version,
        "attia_source_commit": attia_source_commit,
        "n_seeds": n_seeds,
        "budgets": list(budgets) if budgets is not None else None,
        "strategies": strategies,
        "initial_policies": initial_policies,
        "candidate_pool_size": candidate_pool_size,
        "duplicate_tolerance": duplicate_tolerance,
        "n_jobs": n_jobs,
        "reference_underestimated": reference_underestimated,
        "continuous_regret_valid": not reference_underestimated,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if extra_metadata:
        manifest.update(dict(extra_metadata))

    return manifest


@dataclass(frozen=True)
class ScientificModelProvenance:
    """Immutable model run provenance object uniquely identifying a trained scientific model instance."""

    model_run_id: str
    dataset_name: str
    dataset_fingerprint: str
    spec_fingerprint: str
    training_experiment_ids: list[str]
    feature_columns: list[str]
    target_columns: list[str]
    random_seed: int
    model_types: dict[str, str]
    code_head_commit: str | None
    git_dirty: bool
    git_diff_sha256: str | None
    direct_training_experiment_ids: list[str] = field(default_factory=list)
    stage_a_training_experiment_ids_per_channel: dict[str, list[str]] = field(default_factory=dict)
    stage_b_training_experiment_ids_per_target: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    library_versions: dict[str, str] = field(default_factory=get_environment_provenance)

    @classmethod
    def create(
        cls,
        dataset_name: str,
        dataset_fingerprint: str,
        spec_fingerprint: str,
        training_experiment_ids: list[str],
        feature_columns: list[str],
        target_columns: list[str],
        random_seed: int,
        model_types: dict[str, str],
        direct_training_experiment_ids: list[str] | None = None,
        stage_a_training_experiment_ids_per_channel: dict[str, list[str]] | None = None,
        stage_b_training_experiment_ids_per_target: dict[str, list[str]] | None = None,
        repo_root: Path | str | None = None,
    ) -> ScientificModelProvenance:
        git_info = get_git_provenance(repo_root)

        direct_ids = sorted(direct_training_experiment_ids or training_experiment_ids)
        stage_a_ids = {k: sorted(v) for k, v in (stage_a_training_experiment_ids_per_channel or {}).items()}
        stage_b_ids = {k: sorted(v) for k, v in (stage_b_training_experiment_ids_per_target or {}).items()}

        # Deterministic model_run_id from inputs (respecting declared feature order)
        hasher = hashlib.sha256()
        hasher.update(dataset_name.encode("utf-8"))
        hasher.update(dataset_fingerprint.encode("utf-8"))
        hasher.update(spec_fingerprint.encode("utf-8"))
        hasher.update(json.dumps(sorted(training_experiment_ids)).encode("utf-8"))
        hasher.update(json.dumps(direct_ids).encode("utf-8"))
        hasher.update(json.dumps(stage_a_ids, sort_keys=True).encode("utf-8"))
        hasher.update(json.dumps(stage_b_ids, sort_keys=True).encode("utf-8"))
        hasher.update(json.dumps(list(feature_columns)).encode("utf-8"))  # Ordered
        hasher.update(json.dumps(list(target_columns)).encode("utf-8"))   # Ordered
        hasher.update(str(random_seed).encode("utf-8"))
        hasher.update(json.dumps(model_types, sort_keys=True).encode("utf-8"))
        if git_info["code_head_commit"]:
            hasher.update(git_info["code_head_commit"].encode("utf-8"))
        if git_info["git_diff_sha256"]:
            hasher.update(git_info["git_diff_sha256"].encode("utf-8"))

        model_run_id = f"RUN_{hasher.hexdigest()[:16]}"

        return cls(
            model_run_id=model_run_id,
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            spec_fingerprint=spec_fingerprint,
            training_experiment_ids=sorted(training_experiment_ids),
            feature_columns=list(feature_columns),
            target_columns=list(target_columns),
            random_seed=random_seed,
            model_types=dict(model_types),
            code_head_commit=git_info["code_head_commit"],
            git_dirty=git_info["git_dirty"],
            git_diff_sha256=git_info["git_diff_sha256"],
            direct_training_experiment_ids=direct_ids,
            stage_a_training_experiment_ids_per_channel=stage_a_ids,
            stage_b_training_experiment_ids_per_target=stage_b_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

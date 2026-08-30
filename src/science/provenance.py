from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def get_git_provenance(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Extracts git revision, branch, dirty status, and tracked diff hash safely."""
    if repo_root is None:
        repo_path = Path(__file__).resolve().parents[2]
    else:
        repo_path = Path(repo_root)

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
    """
    if df.empty:
        return hashlib.sha256(b"EMPTY_DATASET").hexdigest()[:16]

    cols_to_use = []
    if id_col and id_col in df.columns:
        cols_to_use.append(id_col)
    for c in sorted(feature_cols):
        if c in df.columns and c not in cols_to_use:
            cols_to_use.append(c)
    for c in sorted(target_cols):
        if c in df.columns and c not in cols_to_use:
            cols_to_use.append(c)

    sub_df = df[cols_to_use].copy()

    # Sort deterministically
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
                row_repr.append(f"{float(val):.8e}")
            elif isinstance(val, (int, np.integer)):
                row_repr.append(str(int(val)))
            else:
                row_repr.append(str(val).strip())
        hasher.update(("|".join(row_repr) + "\n").encode("utf-8"))

    return hasher.hexdigest()[:16]


def compute_spec_fingerprint(
    spec: Any,
    two_stage_spec: Any | None = None,
) -> str:
    """Computes a deterministic SHA-256 fingerprint over the scientific dataset and two-stage specs."""
    spec_dict: dict[str, Any] = {
        "name": getattr(spec, "name", ""),
        "pre_experiment_features": sorted(getattr(spec, "pre_experiment_features", [])),
        "candidate_variables": sorted(getattr(spec, "candidate_variables", [])),
        "post_experiment_characterization": sorted(getattr(spec, "post_experiment_characterization", [])),
        "targets": sorted(getattr(spec, "targets", [])),
        "target_column": getattr(spec, "target_column", ""),
        "objective": getattr(spec, "objective", "maximize"),
        "constraints": sorted(getattr(spec, "constraints", [])),
    }
    if two_stage_spec is not None:
        spec_dict["two_stage"] = {
            "process_features": sorted(getattr(two_stage_spec, "process_features", [])),
            "characterization_targets": sorted(getattr(two_stage_spec, "characterization_targets", [])),
            "performance_targets": sorted(getattr(two_stage_spec, "performance_targets", [])),
        }
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
        repo_root: Path | str | None = None,
    ) -> ScientificModelProvenance:
        git_info = get_git_provenance(repo_root)

        # Deterministic model_run_id from inputs
        hasher = hashlib.sha256()
        hasher.update(dataset_name.encode("utf-8"))
        hasher.update(dataset_fingerprint.encode("utf-8"))
        hasher.update(spec_fingerprint.encode("utf-8"))
        hasher.update(json.dumps(sorted(training_experiment_ids)).encode("utf-8"))
        hasher.update(json.dumps(sorted(feature_columns)).encode("utf-8"))
        hasher.update(json.dumps(sorted(target_columns)).encode("utf-8"))
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
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

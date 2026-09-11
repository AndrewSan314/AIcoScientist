from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter
from src.datasets.battery_process.base import ProcessOptimizationTask, ProcessPredictionTask
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.evaluation import reveal_one
from src.process.fusion.encoders import SignalFeatureEncoder
from src.process.models.flat_baseline import GaussianProcessBaseline, TreeEnsembleBaseline
from src.process.models.uncertainty import conformal_interval
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.stages import ProcessStage


ADAPTERS = [DrakopoulosGraphiteAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter, NaIonHTEAdapter, ArtisticSimulationAdapter]
REPLAY_TASKS = {
    "drakopoulos_graphite": ("cell_capacity_mah", ProcessStage.COATING),
    "warwick_nmc622": ("calendered_density_g_cm3", ProcessStage.CALENDERING),
}
OOD_TASKS = {
    "drakopoulos_graphite": ("cell_capacity_mah", "coating.coating_speed_m_per_min"),
    "warwick_nmc622": ("calendered_density_g_cm3", "calendering.roll_gap_um"),
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    residual = actual - predicted
    variance = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.sum(residual ** 2) / variance) if len(actual) > 1 and variance else None,
    }


def _grouped_prediction(adapter, *, seed: int) -> list[dict[str, object]]:
    targets = sorted({name for run in adapter.load_runs() for name, value in run.final_kpis.items() if isinstance(value.value, (int, float))})
    reports: list[dict[str, object]] = []
    for target in targets:
        try:
            frame = adapter.build_training_view(ProcessPredictionTask(target, ProcessStage.FINAL_CHARACTERIZATION))
        except ValueError as exc:
            reports.append({"target": target, "status": "SKIPPED", "reason": str(exc)})
            continue
        X, y, groups = frame.features.to_numpy(), frame.targets.to_numpy(), frame.groups.to_numpy()
        if len(X) < 8 or len(set(groups)) < 4:
            reports.append({"target": target, "status": "SKIPPED", "reason": "requires at least 8 rows and 4 independent groups", "rows": len(X), "groups": len(set(groups))})
            continue
        outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_calibration, test = next(outer.split(X, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
        train_relative, calibration_relative = next(inner.split(X[train_calibration], y[train_calibration], groups[train_calibration]))
        train, calibration = train_calibration[train_relative], train_calibration[calibration_relative]
        model = TreeEnsembleBaseline(random_state=seed).fit(X[train], y[train])
        calibration_mean, _ = model.predict_distribution(X[calibration])
        test_mean, _ = model.predict_distribution(X[test])
        lower, upper = conformal_interval(test_mean, y[calibration] - calibration_mean)
        baselines = [{"model": "ExtraTreesRegressor", "metrics": _metrics(y[test], test_mean)}]
        forest_mean, _ = TreeEnsembleBaseline("random_forest", random_state=seed).fit(X[train], y[train]).predict_distribution(X[test])
        baselines.append({"model": "RandomForestRegressor", "metrics": _metrics(y[test], forest_mean)})
        if X.shape[1] <= 12:
            gp_mean, _ = GaussianProcessBaseline(random_state=seed).fit(X[train], y[train]).predict_distribution(X[test])
            baselines.append({"model": "GaussianProcessRegressor", "metrics": _metrics(y[test], gp_mean)})
        reports.append({
            "target": target, "status": "EVALUATED", "model": "ExtraTreesRegressor", "split": "grouped holdout with disjoint grouped calibration",
            "rows": len(X), "groups": len(set(groups)), "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
            "metrics": _metrics(y[test], test_mean), "baselines": baselines,
            "interval": {"nominal_coverage": 0.9, "empirical_coverage": float(np.mean((y[test] >= lower) & (y[test] <= upper))), "calibration_residual_count": len(calibration)},
        })
    return reports


def _offline_replay(adapter, *, seed: int) -> dict[str, object]:
    task_definition = REPLAY_TASKS.get(adapter.metadata().dataset_id)
    if task_definition is None:
        return {"status": "SKIPPED", "reason": "no audited recipe-level replay target"}
    target, stage = task_definition
    task = ProcessOptimizationTask(target, stage)
    replay = adapter.build_replay_frame(task)
    if len(replay) <= 3:
        return {"status": "SKIPPED", "reason": "fewer than four source recipes"}
    observed, hidden = replay.iloc[:3].copy(), replay.iloc[3:].copy()
    try:
        proposal = ProcessOptimizationCoordinator().propose_recipes(
            observed, adapter.build_optimization_space(task), ProcessOptimizationObjective([ObjectiveSpec(target, "maximize")]), seed=seed,
        )[0]
    except RuntimeError as exc:
        return {"status": "SKIPPED_DEPENDENCY", "reason": str(exc)}
    observed, hidden, revealed = reveal_one(observed, hidden, recipe_id=proposal.source_recipe_id or "", id_column="recipe_id", target=target)
    best_so_far = float(observed[target].max())
    oracle_best = float(replay[target].max())
    return {
        "status": "REVEALED", "target": target, "stage": stage.value, "seed": seed,
        "proposal_recipe_id": proposal.source_recipe_id, "proposal_controls": proposal.controls,
        "revealed_recipe_id": revealed.recipe_id, "revealed_target": revealed.revealed_target,
        "best_so_far": best_so_far, "oracle_best": oracle_best, "simple_regret": max(0.0, oracle_best - best_so_far), "remaining_hidden": len(hidden),
    }


def _ultrasound_ablation(adapter, *, seed: int) -> dict[str, object]:
    if adapter.metadata().dataset_id != "warwick_ultrasound":
        return {"status": "SKIPPED", "reason": "no source-backed paired process/spectrum task"}
    rows = []
    encoder = SignalFeatureEncoder()
    for run in adapter.load_runs():
        target = run.final_kpis.get("post_calendering_thickness_um")
        stage = next((item for item in run.stages if item.stage_type == ProcessStage.CALENDERING), None)
        spectrum = next((item for item in stage.modalities if item.modality_id.endswith("before_spectrum")), None) if stage else None
        if target is None or not isinstance(target.value, (int, float)) or spectrum is None:
            continue
        rows.append((dict(stage.controls), encoder.encode(spectrum.values["fft_magnitude"]), float(target.value), run.batch_id or run.run_id))
    if len(rows) < 8 or len({row[3] for row in rows}) < 4:
        return {"status": "SKIPPED", "reason": "insufficient grouped paired process/spectrum rows"}
    columns = sorted({name for controls, _, _, _ in rows for name in controls})
    process = np.asarray([[float(controls[name].value) if name in controls else 0.0 for name in columns] + [float(name in controls) for name in columns] for controls, _, _, _ in rows])
    signal = np.asarray([features for _, features, _, _ in rows])
    target = np.asarray([value for _, _, value, _ in rows])
    groups = np.asarray([group for _, _, _, group in rows])
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(process, target, groups))
    reports = []
    for name, features in {"process_only": process, "signal_only": signal, "naive_concatenation": np.column_stack((process, signal))}.items():
        prediction, _ = TreeEnsembleBaseline(random_state=seed).fit(features[train], target[train]).predict_distribution(features[test])
        reports.append({"mode": name, "metrics": _metrics(target[test], prediction)})
    return {
        "status": "EVALUATED", "target": "post_calendering_thickness_um", "split": "grouped process-condition holdout",
        "rows": len(rows), "groups": len(set(groups)), "reports": reports,
        "gated_fusion": {"status": "IMPLEMENTED_NOT_CLAIMED", "reason": "GatedMaskedFusion is unit-tested; this small-N benchmark reports auditable scalar baselines without a superiority claim."},
    }


def _stage_ablation(adapter, *, seed: int) -> dict[str, object]:
    """Evaluate only information available by each recorded process stage."""
    targets = sorted({name for run in adapter.load_runs() for name, value in run.final_kpis.items() if isinstance(value.value, (int, float))})
    reports: list[dict[str, object]] = []
    for target in targets:
        for stage in adapter.metadata().process_stages:
            try:
                frame = adapter.build_training_view(ProcessPredictionTask(target, stage))
            except ValueError as exc:
                reports.append({"target": target, "stage": stage.value, "status": "SKIPPED", "reason": str(exc)})
                continue
            X, y, groups = frame.features.to_numpy(), frame.targets.to_numpy(), frame.groups.to_numpy()
            if len(X) < 8 or len(set(groups)) < 4:
                reports.append({"target": target, "stage": stage.value, "status": "SKIPPED", "reason": "requires at least 8 rows and 4 independent groups", "rows": len(X), "groups": len(set(groups))})
                continue
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
            predicted, _ = TreeEnsembleBaseline(random_state=seed).fit(X[train], y[train]).predict_distribution(X[test])
            reports.append({"target": target, "stage": stage.value, "status": "EVALUATED", "model": "ExtraTreesRegressor", "split": "grouped holdout", "rows": len(X), "groups": len(set(groups)), "feature_count": X.shape[1], "metrics": _metrics(y[test], predicted)})
    return {
        "dataset_id": adapter.metadata().dataset_id,
        "reports": reports,
        "stage_transition_model": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "The audited sources do not provide enough linked multi-stage trajectories for a source-backed transition evaluation."},
    }


def _extreme_ood_stress(adapter, *, seed: int) -> dict[str, object]:
    task_definition = OOD_TASKS.get(adapter.metadata().dataset_id)
    if task_definition is None:
        return {"status": "SKIPPED", "reason": "no audited recipe control supports an extreme-condition holdout"}
    target, feature = task_definition
    try:
        frame = adapter.build_training_view(ProcessPredictionTask(target, ProcessStage.FINAL_CHARACTERIZATION))
    except ValueError as exc:
        return {"status": "SKIPPED", "reason": str(exc)}
    if feature not in frame.features:
        return {"status": "SKIPPED", "reason": f"source control {feature!r} is unavailable at this information horizon"}
    values = frame.features[feature].to_numpy(dtype=float)
    held_out = np.isclose(values, values.min()) | np.isclose(values, values.max())
    train, test = np.flatnonzero(~held_out), np.flatnonzero(held_out)
    groups = frame.groups.to_numpy()
    if len(train) < 4 or len(test) < 2 or set(groups[train]) & set(groups[test]):
        return {"status": "SKIPPED", "reason": "source extremes do not produce a disjoint grouped holdout", "train_rows": len(train), "test_rows": len(test)}
    X, y = frame.features.to_numpy(), frame.targets.to_numpy()
    predicted, _ = TreeEnsembleBaseline(random_state=seed).fit(X[train], y[train]).predict_distribution(X[test])
    return {
        "status": "EVALUATED", "target": target, "source_control": feature,
        "split": "minimum-and-maximum source-control values held out as OOD recipes",
        "process_dimension": frame.features.shape[1], "candidate_pool_size": len(frame.features),
        "train_rows": len(train), "test_rows": len(test), "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])),
        "metrics": _metrics(y[test], predicted),
        "missing_intermediate_observations": {"status": "NOT_EVALUATED", "reason": "the parsed source subset lacks source-linked repeated intermediate observations per recipe"},
        "multiobjective_frontier": {"status": "NOT_EVALUATED", "reason": "no prespecified, source-backed multiobjective target pair is registered for this replay task"},
    }


def _write_prediction_figure(predictions: list[dict[str, object]], output: Path) -> None:
    rows = [
        (f"{report['dataset_id']}\n{item['target']}", item["metrics"]["r2"])
        for report in predictions for item in report["reports"]
        if item["status"] == "EVALUATED" and item["metrics"]["r2"] is not None
    ]
    if not rows:
        (output / "README.md").write_text("No comparable grouped-holdout R² values were available to plot.\n", encoding="utf-8")
        return
    import matplotlib.pyplot as plt

    labels, scores = zip(*rows)
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    axis.bar(range(len(scores)), scores, color="#2b6cb0")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_ylabel("Grouped holdout R²")
    axis.set_title("BPSS prediction results (not a cross-chemistry ranking)")
    figure.tight_layout()
    figure.savefig(output / "grouped_prediction_r2.png", dpi=180)
    plt.close(figure)


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_report(root: Path, manifest: dict[str, object]) -> None:
    command = " ".join(sys.argv)
    unavailable = manifest["unavailable"] or ["none"]
    (root / "PROCESS_BENCHMARK_REPORT.md").write_text(
        "# Battery Process Stress Suite\n\n"
        f"Status: **{manifest['status']}**.\n\n"
        "## Reproducibility\n\n"
        f"- Command: `{command}`\n"
        f"- Commit: `{_git_revision()}`\n"
        f"- Python: `{sys.version.split()[0]}`\n"
        "- Process tests (run separately): `python -m pytest -q tests/process -p no:cacheprovider`\n"
        "- Artifact manifest: `manifest.json`\n\n"
        "## Limitations\n\n"
        "- Small, source-specific datasets; no chemistry-generalization claim.\n"
        "- Missing modalities are omitted rather than imputed.\n"
        f"- Unavailable raw sources: {', '.join(unavailable)}.\n"
        "- No live production control or causal-effect claim.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-backed BPSS benchmark manifest.")
    parser.add_argument("--output", type=Path, default=Path("outputs/process_benchmark"))
    parser.add_argument("--allow-unavailable", action="store_true", help="Write an audit-only manifest when raw source data have not been acquired.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-seeds", type=int, default=3)
    args = parser.parse_args()
    root = args.output
    for directory in ("dataset_audits", "prediction", "calibration", "multimodal_ablations", "missing_modality", "stage_ablations", "optimization", "stress", "figures"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    audits = []
    unavailable = []
    evaluated = []
    replayed = []
    prediction_artifacts = []
    for adapter_class in ADAPTERS:
        adapter = adapter_class()
        report = adapter.validate()
        audit = {"metadata": adapter.metadata().__dict__, "validation": {"valid": report.valid, "errors": list(report.errors)}}
        dataset_id = adapter.metadata().dataset_id
        (root / "dataset_audits" / f"{dataset_id}.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        audits.append(audit)
        if not report.valid:
            unavailable.append(dataset_id)
            blocked = {
                "dataset_id": dataset_id, "status": "BLOCKED_SOURCE_ACCESS", "reason": list(report.errors),
                "simulation_manifest": {"status": "NOT_GENERATED", "reason": "official simulator raw files require source access"},
            }
            (root / "stage_ablations" / f"{dataset_id}.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
            (root / "stress" / f"{dataset_id}.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
            continue
        predictions = _grouped_prediction(adapter, seed=args.seed)
        prediction_report = {"dataset_id": dataset_id, "evidence_kind": adapter.metadata().evidence_kind, "reports": predictions}
        (root / "prediction" / f"{dataset_id}.json").write_text(json.dumps(prediction_report, indent=2), encoding="utf-8")
        prediction_artifacts.append(prediction_report)
        (root / "calibration" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "reports": [{"target": item["target"], "status": item["status"], "interval": item.get("interval")} for item in predictions]}, indent=2), encoding="utf-8")
        ablation = _ultrasound_ablation(adapter, seed=args.seed)
        (root / "multimodal_ablations" / f"{dataset_id}.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
        (root / "missing_modality" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "status": ablation["status"], "policy": "An unavailable modality is omitted, never zero-filled.", "process_only_reference": next((item for item in ablation.get("reports", []) if item["mode"] == "process_only"), None)}, indent=2), encoding="utf-8")
        (root / "stage_ablations" / f"{dataset_id}.json").write_text(json.dumps(_stage_ablation(adapter, seed=args.seed), indent=2), encoding="utf-8")
        (root / "stress" / f"{dataset_id}.json").write_text(json.dumps(_extreme_ood_stress(adapter, seed=args.seed), indent=2), encoding="utf-8")
        replays = [_offline_replay(adapter, seed=args.seed + offset) for offset in range(args.replay_seeds)]
        (root / "optimization" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "replays": replays}, indent=2), encoding="utf-8")
        evaluated.append(dataset_id)
        if any(replay["status"] == "REVEALED" for replay in replays):
            replayed.append(dataset_id)
    manifest = {"suite": "BPSS", "datasets": audits, "status": "PARTIAL" if unavailable else "READY", "unavailable": unavailable, "evaluated": evaluated, "replayed": replayed, "seed": args.seed, "replay_seeds": args.replay_seeds}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _write_prediction_figure(prediction_artifacts, root / "figures")
    _write_report(root, manifest)
    if unavailable and not args.allow_unavailable:
        raise SystemExit("BPSS refused to benchmark unaudited source data; rerun with --allow-unavailable for an audit-only manifest.")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

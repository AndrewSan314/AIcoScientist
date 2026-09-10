from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter
from src.datasets.battery_process.base import ProcessOptimizationTask, ProcessPredictionTask
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.evaluation import reveal_one
from src.process.models.flat_baseline import TreeEnsembleBaseline
from src.process.models.uncertainty import conformal_interval
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.stages import ProcessStage


ADAPTERS = [DrakopoulosGraphiteAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter, NaIonHTEAdapter, ArtisticSimulationAdapter]
REPLAY_TASKS = {
    "drakopoulos_graphite": ("cell_capacity_mah", ProcessStage.COATING),
    "warwick_nmc622": ("calendered_density_g_cm3", ProcessStage.CALENDERING),
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
        reports.append({
            "target": target, "status": "EVALUATED", "model": "ExtraTreesRegressor", "split": "grouped holdout with disjoint grouped calibration",
            "rows": len(X), "groups": len(set(groups)), "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
            "metrics": _metrics(y[test], test_mean), "interval": {"nominal_coverage": 0.9, "empirical_coverage": float(np.mean((y[test] >= lower) & (y[test] <= upper))), "calibration_residual_count": len(calibration)},
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
    return {
        "status": "REVEALED", "target": target, "stage": stage.value, "seed": seed,
        "proposal_recipe_id": proposal.source_recipe_id, "proposal_controls": proposal.controls,
        "revealed_recipe_id": revealed.recipe_id, "revealed_target": revealed.revealed_target,
        "best_so_far": float(observed[target].max()), "remaining_hidden": len(hidden),
    }


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
    for adapter_class in ADAPTERS:
        adapter = adapter_class()
        report = adapter.validate()
        audit = {"metadata": adapter.metadata().__dict__, "validation": {"valid": report.valid, "errors": list(report.errors)}}
        dataset_id = adapter.metadata().dataset_id
        (root / "dataset_audits" / f"{dataset_id}.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        audits.append(audit)
        if not report.valid:
            unavailable.append(dataset_id)
            continue
        predictions = _grouped_prediction(adapter, seed=args.seed)
        prediction_report = {"dataset_id": dataset_id, "evidence_kind": adapter.metadata().evidence_kind, "reports": predictions}
        (root / "prediction" / f"{dataset_id}.json").write_text(json.dumps(prediction_report, indent=2), encoding="utf-8")
        (root / "calibration" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "reports": [{"target": item["target"], "status": item["status"], "interval": item.get("interval")} for item in predictions]}, indent=2), encoding="utf-8")
        replays = [_offline_replay(adapter, seed=args.seed + offset) for offset in range(args.replay_seeds)]
        (root / "optimization" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "replays": replays}, indent=2), encoding="utf-8")
        evaluated.append(dataset_id)
        if any(replay["status"] == "REVEALED" for replay in replays):
            replayed.append(dataset_id)
    manifest = {"suite": "BPSS", "datasets": audits, "status": "PARTIAL" if unavailable else "READY", "unavailable": unavailable, "evaluated": evaluated, "replayed": replayed, "seed": args.seed, "replay_seeds": args.replay_seeds}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (root / "PROCESS_BENCHMARK_REPORT.md").write_text(
        "# Battery Process Stress Suite\n\n" + "Grouped ExtraTrees/split-conformal artifacts and no-lookahead recipe replays are written for supported source-backed tasks. " + ("Unavailable sources: " + ", ".join(unavailable) if unavailable else "All registered adapters passed source validation."),
        encoding="utf-8",
    )
    if unavailable and not args.allow_unavailable:
        raise SystemExit("BPSS refused to benchmark unaudited source data; rerun with --allow-unavailable for an audit-only manifest.")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

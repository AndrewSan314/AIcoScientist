from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter
from src.datasets.battery_process.base import ProcessOptimizationTask, ProcessPredictionTask
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.evaluation import reveal_one
from src.process.fusion import CrossAttentionSetFusion, GatedMaskedFusion
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
REPLAY_STRATEGIES = ("random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement", "thompson")
CAPABILITY_SECTIONS = {
    "multiobjective": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "Official qNEHVI support is unit-tested, but no registered BPSS task has a prespecified source-backed objective pair and reference point."},
    "constraints": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "Hard scalar constraint semantics are unit-tested; no BPSS replay currently registers a source-backed modeled outcome constraint."},
    "manufacturability": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "The learned feasibility model exists, but BPSS adapters do not expose an audited process-failure label set for evaluation."},
    "evidence_acquisition": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "Three pre-reveal policies are unit-tested, but no BPSS adapter registers multiple legal source-backed evidence options at one decision horizon."},
    "multifidelity": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "Paired low-to-high residual correction is unit-tested and rejects unmatched/incompatible pairs; no compatible paired BPSS evidence is available for evaluation."},
    "robustness": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "Robustness utilities are available, but no source-backed perturbation protocol is registered for these BPSS tasks."},
    "explainability": {"status": "IMPLEMENTED_NOT_EVALUATED", "reason": "Stage ablations and process-graph path attribution are available; no source-backed local-recommendation attribution task is registered."},
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
            reports.append({"target": target, "status": "NOT_EVALUATED", "reason": str(exc)})
            continue
        X, y, groups = frame.features.to_numpy(), frame.targets.to_numpy(), frame.groups.to_numpy()
        if len(X) < 8 or len(set(groups)) < 4:
            reports.append({"target": target, "status": "NOT_EVALUATED", "reason": "requires at least 8 rows and 4 independent groups", "rows": len(X), "groups": len(set(groups))})
            continue
        outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_calibration, test = next(outer.split(X, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
        train_relative, calibration_relative = next(inner.split(X[train_calibration], y[train_calibration], groups[train_calibration]))
        train, calibration = train_calibration[train_relative], train_calibration[calibration_relative]
        model = TreeEnsembleBaseline(random_state=seed).fit(X[train], y[train])
        calibration_mean, _ = model.predict_distribution(X[calibration])
        test_mean, _ = model.predict_distribution(X[test])
        calibration_residuals = y[calibration] - calibration_mean
        intervals = {}
        for coverage in (0.5, 0.8, 0.9, 0.95):
            lower, upper = conformal_interval(test_mean, calibration_residuals, coverage)
            intervals[str(int(coverage * 100))] = {
                "nominal_coverage": coverage,
                "empirical_coverage": float(np.mean((y[test] >= lower) & (y[test] <= upper))),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
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
            "interval": {**intervals["90"], "calibration_residual_count": len(calibration)},
            "intervals": intervals,
        })
    return reports


def _offline_replay(adapter, *, seed: int, strategy: str, max_steps: int) -> dict[str, object]:
    task_definition = REPLAY_TASKS.get(adapter.metadata().dataset_id)
    if task_definition is None:
        return {"status": "NOT_EVALUATED", "strategy": strategy, "reason": "no audited recipe-level replay target"}
    target, stage = task_definition
    task = ProcessOptimizationTask(target, stage)
    replay = adapter.build_replay_frame(task)
    if len(replay) <= 3:
        return {"status": "NOT_EVALUATED", "strategy": strategy, "reason": "fewer than four source recipes"}
    observed, hidden = replay.iloc[:3].copy(), replay.iloc[3:].copy()
    oracle_best = float(replay[target].max())
    trajectory = []
    warmup_started = time.perf_counter()
    try:
        ProcessOptimizationCoordinator().propose_recipes(
            observed, adapter.build_optimization_space(task), ProcessOptimizationObjective([ObjectiveSpec(target, "maximize")]), seed=seed, strategy=strategy,
        )
    except RuntimeError as exc:
        return {"status": "BLOCKED_EXTERNAL", "strategy": strategy, "reason": str(exc), "trajectory": trajectory}
    warmup_latency = time.perf_counter() - warmup_started
    for step in range(min(max_steps, len(hidden))):
        started = time.perf_counter()
        try:
            proposal = ProcessOptimizationCoordinator().propose_recipes(
                observed, adapter.build_optimization_space(task), ProcessOptimizationObjective([ObjectiveSpec(target, "maximize")]), seed=seed + step, strategy=strategy,
            )[0]
        except RuntimeError as exc:
            return {"status": "BLOCKED_EXTERNAL", "strategy": strategy, "reason": str(exc), "trajectory": trajectory}
        latency = time.perf_counter() - started
        observed, hidden, revealed = reveal_one(observed, hidden, recipe_id=proposal.source_recipe_id or "", id_column="recipe_id", target=target)
        best_so_far = float(observed[target].max())
        trajectory.append({
            "decision": step + 1, "proposal_recipe_id": proposal.source_recipe_id, "proposal_controls": proposal.controls,
            "revealed_recipe_id": revealed.recipe_id, "revealed_target": revealed.revealed_target,
            "best_so_far": best_so_far, "simple_regret": max(0.0, oracle_best - best_so_far), "decision_latency_seconds": latency,
        })
    latencies = np.asarray([item["decision_latency_seconds"] for item in trajectory], dtype=float)
    return {
        "status": "EVALUATED", "target": target, "stage": stage.value, "seed": seed, "strategy": strategy,
        "trajectory": trajectory, "best_so_far": trajectory[-1]["best_so_far"], "oracle_best": oracle_best,
        "simple_regret": trajectory[-1]["simple_regret"], "remaining_hidden": len(hidden), "process_decisions": len(trajectory),
        "candidate_pool_size": len(replay), "warmup_decision_latency_seconds": warmup_latency,
        "total_decision_latency_seconds": float(latencies.sum()), "p50_decision_latency_seconds": float(np.quantile(latencies, 0.5)), "p95_decision_latency_seconds": float(np.quantile(latencies, 0.95)), "p99_decision_latency_seconds": float(np.quantile(latencies, 0.99)),
    }


def _latency_report(replays: list[dict[str, object]]) -> dict[str, object]:
    completed = [replay for replay in replays if replay.get("status") == "EVALUATED"]
    decisions = [item["decision_latency_seconds"] for replay in completed for item in replay["trajectory"]]
    if not decisions:
        return {"status": "NOT_EVALUATED", "reason": "no source-backed offline decision replay completed"}
    values = np.asarray(decisions, dtype=float)
    warmups = np.asarray([replay["warmup_decision_latency_seconds"] for replay in completed], dtype=float)
    return {
        "status": "EVALUATED",
        "scope": "Offline source-backed optimizer proposal latency; excludes raw I/O and hardware control.",
        "hardware": {"os": platform.platform(), "processor": platform.processor() or "unknown", "python": sys.version.split()[0]},
        "candidate_pool_sizes": sorted({replay["candidate_pool_size"] for replay in completed}), "batch_size": 1,
        "model_artifact_fingerprint": None, "model_artifact_note": "Offline replay fits the selected surrogate per decision; no frozen artifact is used.",
        "warmup": {"count": len(warmups), "p50_ms": float(np.quantile(warmups, 0.5) * 1000), "p95_ms": float(np.quantile(warmups, 0.95) * 1000), "p99_ms": float(np.quantile(warmups, 0.99) * 1000)},
        "optimizer_proposal": {"count": len(values), "p50_ms": float(np.quantile(values, 0.5) * 1000), "p95_ms": float(np.quantile(values, 0.95) * 1000), "p99_ms": float(np.quantile(values, 0.99) * 1000)},
        "unmeasured_components": ["modality_encoders", "fusion", "stage_state_model", "evidence_selection"],
    }


class _GatedFusionRegressor(nn.Module):
    """Small source-benchmark head composed from the shared missing-aware fusion block."""

    def __init__(self, process_dim: int, signal_dim: int, embedding_dim: int = 4) -> None:
        super().__init__()
        self.process = nn.Linear(process_dim, embedding_dim)
        self.signal = nn.Linear(signal_dim, embedding_dim)
        self.fusion = GatedMaskedFusion(embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, process: torch.Tensor, signal: torch.Tensor, available: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = {"process": torch.tanh(self.process(process))}
        if available["signal"].bool().any():
            tokens["signal"] = torch.tanh(self.signal(signal))
        return self.head(self.fusion(tokens, available)).squeeze(-1)


class _CrossAttentionFusionRegressor(_GatedFusionRegressor):
    """Experimental learned-query set-fusion head for the same source-backed inputs."""

    def __init__(self, process_dim: int, signal_dim: int, embedding_dim: int = 4) -> None:
        super().__init__(process_dim, signal_dim, embedding_dim)
        self.fusion = CrossAttentionSetFusion(embedding_dim, ("process", "signal"))


def _fit_fusion(model_type: type[_GatedFusionRegressor], process: np.ndarray, signal: np.ndarray, target: np.ndarray, *, seed: int) -> tuple[_GatedFusionRegressor, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]:
    process_mean, process_std = process.mean(axis=0), process.std(axis=0)
    signal_mean, signal_std = signal.mean(axis=0), signal.std(axis=0)
    process_std[process_std == 0] = 1.0; signal_std[signal_std == 0] = 1.0
    process_scaled = (process - process_mean) / process_std
    signal_scaled = (signal - signal_mean) / signal_std
    target_mean, target_std = float(target.mean()), float(target.std()) or 1.0
    torch.manual_seed(seed)
    model = model_type(process.shape[1], signal.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.01)
    inputs = torch.tensor(process_scaled, dtype=torch.float32), torch.tensor(signal_scaled, dtype=torch.float32)
    labels = torch.tensor((target - target_mean) / target_std, dtype=torch.float32)
    for epoch in range(200):
        optimizer.zero_grad()
        # Derived stress only: source signal tokens remain absent for masked rows.
        available = {"process": torch.ones(len(target), dtype=torch.bool), "signal": torch.tensor([(index + epoch) % 4 != 0 for index in range(len(target))])}
        loss = torch.mean((model(*inputs, available) - labels) ** 2)
        loss.backward(); optimizer.step()
    return model.eval(), (process_mean, process_std, signal_mean, signal_std, target_mean, target_std)


def _fit_gated_fusion(process: np.ndarray, signal: np.ndarray, target: np.ndarray, *, seed: int) -> tuple[_GatedFusionRegressor, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]:
    return _fit_fusion(_GatedFusionRegressor, process, signal, target, seed=seed)


def _fit_cross_attention_fusion(process: np.ndarray, signal: np.ndarray, target: np.ndarray, *, seed: int) -> tuple[_GatedFusionRegressor, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]:
    return _fit_fusion(_CrossAttentionFusionRegressor, process, signal, target, seed=seed)


def _predict_gated_fusion(model: _GatedFusionRegressor, process: np.ndarray, signal: np.ndarray, scale: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float], *, signal_available: bool | np.ndarray = True) -> np.ndarray:
    process_mean, process_std, signal_mean, signal_std, target_mean, target_std = scale
    if isinstance(signal_available, np.ndarray) and signal_available.shape != (len(process),):
        raise ValueError("signal availability must match the prediction rows")
    available = torch.as_tensor(signal_available, dtype=torch.bool) if isinstance(signal_available, np.ndarray) else torch.full((len(process),), signal_available, dtype=torch.bool)
    with torch.no_grad():
        normalized = model(
            torch.tensor((process - process_mean) / process_std, dtype=torch.float32),
            torch.tensor((signal - signal_mean) / signal_std, dtype=torch.float32),
            {"process": torch.ones(len(process), dtype=torch.bool), "signal": available},
        ).numpy()
    return normalized * target_std + target_mean


def _ultrasound_ablation(adapter, *, seed: int) -> dict[str, object]:
    if adapter.metadata().dataset_id != "warwick_ultrasound":
        return {"status": "NOT_EVALUATED", "reason": "no source-backed paired process/spectrum task"}
    rows = []
    encoder = SignalFeatureEncoder()
    for run in adapter.load_runs():
        target = run.final_kpis.get("post_calendering_thickness_um")
        stage = next((item for item in run.stages if item.stage_type == ProcessStage.CALENDERING), None)
        spectrum = next((item for item in stage.modalities if item.modality_id.endswith("before_spectrum")), None) if stage else None
        if target is None or not isinstance(target.value, (int, float)) or spectrum is None:
            continue
        rows.append((dict(stage.controls), encoder.encode(spectrum.values["fft_magnitude"]), float(target.value), run.batch_id or run.run_id, run.run_id, run.identity_fingerprint))
    if len(rows) < 8 or len({row[3] for row in rows}) < 4:
        return {"status": "NOT_EVALUATED", "reason": "insufficient grouped paired process/spectrum rows"}
    columns = sorted({name for controls, _, _, _, _, _ in rows for name in controls})
    process = np.asarray([[float(controls[name].value) if name in controls else 0.0 for name in columns] + [float(name in controls) for name in columns] for controls, _, _, _, _, _ in rows])
    signal = np.asarray([features for _, features, _, _, _, _ in rows])
    target = np.asarray([value for _, _, value, _, _, _ in rows])
    groups = np.asarray([group for _, _, _, group, _, _ in rows])
    run_ids = np.asarray([run_id for _, _, _, _, run_id, _ in rows])
    source_fingerprint = hashlib.sha256(json.dumps(sorted((run_id, fingerprint) for _, _, _, _, run_id, fingerprint in rows), separators=(",", ":")).encode()).hexdigest()
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(process, target, groups))
    reports = []
    for name, features in {"process_only": process, "signal_only": signal, "naive_concatenation": np.column_stack((process, signal))}.items():
        prediction, _ = TreeEnsembleBaseline(random_state=seed).fit(features[train], target[train]).predict_distribution(features[test])
        reports.append({"mode": name, "metrics": _metrics(target[test], prediction)})
    model, scale = _fit_gated_fusion(process[train], signal[train], target[train], seed=seed)
    reports.append({"mode": "gated_missing_aware_fusion", "metrics": _metrics(target[test], _predict_gated_fusion(model, process[test], signal[test], scale))})
    cross_attention, cross_scale = _fit_cross_attention_fusion(process[train], signal[train], target[train], seed=seed)
    reports.append({"mode": "cross_attention_set_fusion", "experimental": True, "metrics": _metrics(target[test], _predict_gated_fusion(cross_attention, process[test], signal[test], cross_scale))})
    full_drop_ids = sorted(run_ids[test].tolist())
    full_transform = {"kind": "ultrasound_availability_mask", "source_fingerprint": source_fingerprint, "seed": seed, "dropped_source_run_ids": full_drop_ids}
    reports.append({"mode": "gated_fusion_signal_dropout", "derived_stress": True, "evidence_kind": "SIMULATED_STRESS", "stress_transform_fingerprint": hashlib.sha256(json.dumps(full_transform, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), **full_transform, "metrics": _metrics(target[test], _predict_gated_fusion(model, process[test], signal[test], scale, signal_available=False))})
    rng = np.random.default_rng(seed)
    for rate in (0.10, 0.25, 0.50, 0.75):
        unavailable = rng.permutation(len(test))[:max(1, round(rate * len(test)))]
        available = np.ones(len(test), dtype=bool)
        available[unavailable] = False
        transform = {"kind": "ultrasound_availability_mask", "source_fingerprint": source_fingerprint, "seed": seed, "dropped_source_run_ids": sorted(run_ids[test[unavailable]].tolist()), "requested_dropout_rate": rate}
        reports.append({"mode": "gated_fusion_partial_signal_dropout", "derived_stress": True, "evidence_kind": "SIMULATED_STRESS", "missing_modality": "ultrasound", "requested_dropout_rate": rate, "realized_dropout_rate": float((~available).mean()), "stress_transform_fingerprint": hashlib.sha256(json.dumps(transform, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), **transform, "metrics": _metrics(target[test], _predict_gated_fusion(model, process[test], signal[test], scale, signal_available=available))})
    return {
        "status": "EVALUATED", "target": "post_calendering_thickness_um", "split": "grouped process-condition holdout",
        "rows": len(rows), "groups": len(set(groups)), "source_fingerprint": source_fingerprint, "reports": reports,
        "gated_fusion": {"status": "EVALUATED", "model": "GatedMaskedFusion with train-only standardized source descriptors", "reason": "Small grouped holdout; report only, without a superiority claim."},
        "cross_attention": {"status": "EVALUATED", "model": "CrossAttentionSetFusion with learned query and modality-type embeddings", "reason": "Experimental small-data comparator; report only, without a superiority claim."},
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
                reports.append({"target": target, "stage": stage.value, "status": "NOT_EVALUATED", "reason": str(exc)})
                continue
            X, y, groups = frame.features.to_numpy(), frame.targets.to_numpy(), frame.groups.to_numpy()
            if len(X) < 8 or len(set(groups)) < 4:
                reports.append({"target": target, "stage": stage.value, "status": "NOT_EVALUATED", "reason": "requires at least 8 rows and 4 independent groups", "rows": len(X), "groups": len(set(groups))})
                continue
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
            predicted, _ = TreeEnsembleBaseline(random_state=seed).fit(X[train], y[train]).predict_distribution(X[test])
            reports.append({"target": target, "stage": stage.value, "status": "EVALUATED", "model": "ExtraTreesRegressor", "split": "grouped holdout", "rows": len(X), "groups": len(set(groups)), "feature_count": X.shape[1], "metrics": _metrics(y[test], predicted)})
    return {
        "dataset_id": adapter.metadata().dataset_id,
        "reports": reports,
        "stage_transition_model": {"status": "IMPLEMENTED_NOT_VALIDATED", "reason": "The audited sources do not provide enough linked multi-stage trajectories for a source-backed transition evaluation."},
    }


def _extreme_ood_stress(adapter, *, seed: int) -> dict[str, object]:
    task_definition = OOD_TASKS.get(adapter.metadata().dataset_id)
    if task_definition is None:
        return {"status": "NOT_EVALUATED", "reason": "no audited recipe control supports an extreme-condition holdout"}
    target, feature = task_definition
    try:
        frame = adapter.build_training_view(ProcessPredictionTask(target, ProcessStage.FINAL_CHARACTERIZATION))
    except ValueError as exc:
        return {"status": "NOT_EVALUATED", "reason": str(exc)}
    if feature not in frame.features:
        return {"status": "NOT_EVALUATED", "reason": f"source control {feature!r} is unavailable at this information horizon"}
    values = frame.features[feature].to_numpy(dtype=float)
    held_out = np.isclose(values, values.min()) | np.isclose(values, values.max())
    train, test = np.flatnonzero(~held_out), np.flatnonzero(held_out)
    groups = frame.groups.to_numpy()
    if len(train) < 4 or len(test) < 2 or set(groups[train]) & set(groups[test]):
        return {"status": "NOT_EVALUATED", "reason": "source extremes do not produce a disjoint grouped holdout", "train_rows": len(train), "test_rows": len(test)}
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
    architecture_status = manifest["architecture_status"]
    fusion_path = root / "multimodal_ablations" / "warwick_ultrasound.json"
    fusion_lines = ["- Warwick ultrasound: NOT_EVALUATED."]
    if fusion_path.exists():
        fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        reports = {item["mode"]: item for item in fusion.get("reports", [])}
        if fusion.get("status") == "EVALUATED" and "gated_missing_aware_fusion" in reports:
            gated = reports["gated_missing_aware_fusion"]["metrics"]["r2"]
            naive = reports.get("naive_concatenation", {}).get("metrics", {}).get("r2")
            cross = reports.get("cross_attention_set_fusion", {}).get("metrics", {}).get("r2")
            rates = [item["requested_dropout_rate"] for item in fusion.get("reports", []) if item.get("mode") == "gated_fusion_partial_signal_dropout"]
            stress = f" Derived ultrasound-masking stress rates: {', '.join(f'{rate:.0%}' for rate in rates)}." if rates else ""
            experimental = f" Experimental cross-attention R²={cross:.3f}." if cross is not None else ""
            fusion_lines = [f"- Warwick ultrasound grouped holdout ({fusion['rows']} rows/{fusion['groups']} groups): gated R²={gated:.3f}; naive concatenation R²={naive:.3f}. Gated fusion is not superior on this split.{experimental}{stress}"]
    replay_lines = []
    for path in sorted((root / "optimization").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for strategy in sorted({item.get("strategy") for item in result.get("replays", []) if item.get("status") == "EVALUATED"}):
            regrets = [item["simple_regret"] for item in result["replays"] if item.get("status") == "EVALUATED" and item.get("strategy") == strategy]
            replay_lines.append(f"- {result['dataset_id']} {strategy}: {len(regrets)} seeds, mean final simple regret={sum(regrets) / len(regrets):.6g}.")
    (root / "PROCESS_BENCHMARK_REPORT.md").write_text(
        "# Battery Process Stress Suite\n\n"
        f"Status: **{manifest['status']}**.\n\n"
        "## Architecture status\n\n"
        + "\n".join(f"- {name}: **{item['status']}** — {item['reason']}" for name, item in architecture_status.items())
        + "\n\n"
        + "## Source-backed results\n\n"
        + "\n".join(fusion_lines)
        + "\n\n"
        + ("## Offline replay\n\n" + "\n".join(replay_lines) + "\n\n" if replay_lines else "")
        + "## Reproducibility\n\n"
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
    parser.add_argument("--replay-strategies", choices=REPLAY_STRATEGIES, nargs="+", default=REPLAY_STRATEGIES)
    parser.add_argument("--replay-steps", type=int, default=3)
    args = parser.parse_args()
    if args.replay_steps < 1:
        parser.error("--replay-steps must be positive")
    root = args.output
    for directory in ("dataset_audits", "prediction", "calibration", "multimodal_ablations", "missing_modality", "stage_ablations", "optimization", "stress", "latency", "figures", *CAPABILITY_SECTIONS):
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
            artistic_pending = dataset_id == "artistic"
            blocked = {
                "dataset_id": dataset_id,
                "status": "BLOCKED_EXTERNAL",
                "reason": (["Pinned public ARTISTIC source is available, but no valid real LAMMPS execution has been normalized; simulated data are not fabricated."] if artistic_pending else []) + list(report.errors),
                "simulation_manifest": {
                    "status": "IMPLEMENTED_NOT_VALIDATED" if artistic_pending else "NOT_EVALUATED",
                    "reason": "a full source-backed ARTISTIC execution is required before model fitting" if artistic_pending else "source files require audit",
                },
            }
            (root / "stage_ablations" / f"{dataset_id}.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
            (root / "stress" / f"{dataset_id}.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
            (root / "latency" / f"{dataset_id}.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
            continue
        predictions = _grouped_prediction(adapter, seed=args.seed)
        prediction_report = {"dataset_id": dataset_id, "evidence_kind": adapter.metadata().evidence_kind, "reports": predictions}
        (root / "prediction" / f"{dataset_id}.json").write_text(json.dumps(prediction_report, indent=2), encoding="utf-8")
        prediction_artifacts.append(prediction_report)
        (root / "calibration" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "reports": [{"target": item["target"], "status": item["status"], "interval": item.get("interval"), "intervals": item.get("intervals")} for item in predictions]}, indent=2), encoding="utf-8")
        ablation = _ultrasound_ablation(adapter, seed=args.seed)
        (root / "multimodal_ablations" / f"{dataset_id}.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
        (root / "missing_modality" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "status": ablation["status"], "policy": "An unavailable modality is omitted, never zero-filled.", "process_only_reference": next((item for item in ablation.get("reports", []) if item["mode"] == "process_only"), None)}, indent=2), encoding="utf-8")
        (root / "stage_ablations" / f"{dataset_id}.json").write_text(json.dumps(_stage_ablation(adapter, seed=args.seed), indent=2), encoding="utf-8")
        (root / "stress" / f"{dataset_id}.json").write_text(json.dumps(_extreme_ood_stress(adapter, seed=args.seed), indent=2), encoding="utf-8")
        replays = [_offline_replay(adapter, seed=args.seed + offset, strategy=strategy, max_steps=args.replay_steps) for strategy in args.replay_strategies for offset in range(args.replay_seeds)]
        (root / "optimization" / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "replays": replays}, indent=2), encoding="utf-8")
        (root / "latency" / f"{dataset_id}.json").write_text(json.dumps(_latency_report(replays), indent=2), encoding="utf-8")
        evaluated.append(dataset_id)
        if any(replay["status"] == "EVALUATED" for replay in replays):
            replayed.append(dataset_id)
    manifest = {
        "suite": "BPSS", "datasets": audits, "status": "PARTIAL" if unavailable else "READY", "unavailable": unavailable,
        "evaluated": evaluated, "replayed": replayed, "seed": args.seed, "replay_seeds": args.replay_seeds, "replay_strategies": args.replay_strategies, "replay_steps": args.replay_steps,
        "architecture_status": {
            "scalar_contextual_stage_optimizer": {
                "status": "IMPLEMENTED_NOT_VALIDATED",
                "reason": "Context-conditioned stage-wise optimizer is implemented and unit-tested; source-backed multi-stage trajectory validation remains unavailable.",
            },
            "multimodal_contextual_state": {
                "status": "IMPLEMENTED_NOT_VALIDATED",
                "reason": "Multimodal latent state adapter is implemented and unit-tested; no source-backed trained state benchmark is claimed.",
            },
            "horizon_safe_evidence_replay": {
                "status": "IMPLEMENTED_NOT_VALIDATED",
                "reason": "Horizon-safe source-backed evidence replay is implemented and unit-tested; no adaptive policy performance is claimed.",
            },
            "adaptive_evidence_policy": {
                "status": "IMPLEMENTED_NOT_VALIDATED",
                "reason": "Three pre-reveal evidence policies and blinded source-backed reveal are implemented and unit-tested; no source-backed policy-performance or EVI claim is made.",
            },
        },
        "capability_sections": CAPABILITY_SECTIONS,
    }
    for section, status in CAPABILITY_SECTIONS.items():
        (root / section / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _write_prediction_figure(prediction_artifacts, root / "figures")
    _write_report(root, manifest)
    if unavailable and not args.allow_unavailable:
        raise SystemExit("BPSS refused to benchmark unaudited source data; rerun with --allow-unavailable for an audit-only manifest.")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

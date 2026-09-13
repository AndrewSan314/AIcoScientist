from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace

from .core import DatasetAdapter, ProcessSurrogate, ProcessSurrogateSample, SurrogateArtifact, SurrogateDecisionContext, SurrogateInputSchema, TrainOnlyPreprocessor, evaluate, split_groups


@dataclass(frozen=True)
class PipelineConfig:
    targets: tuple[str, ...]
    model_type: str = "gp"
    seed: int = 42
    validation_fraction: float = .2
    test_fraction: float = .2
    objective_target: str | None = None
    objective_sense: str = "maximize"
    proposal_count: int = 3
    exploration_beta: float = 1.
    target_units: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.targets or self.model_type not in {"gp", "extra_trees"} or self.objective_sense not in {"maximize", "minimize"}: raise ValueError("targets, model type, and objective sense are invalid")
        if self.objective_target is not None and self.objective_target not in self.targets: raise ValueError("objective target must be one of the fitted targets")


class ArtifactOptimizerBackend:
    """Existing MASPO backend seam, bound to one immutable decision context."""
    name = "frozen_process_surrogate"
    version = "2"

    def __init__(self, artifact: SurrogateArtifact, context: SurrogateDecisionContext, *, beta: float = 1.) -> None:
        artifact.verify_integrity()
        if context.stage.value != artifact.input_schema.supported_stage: raise ValueError("optimizer decision context stage is unsupported")
        self.artifact, self.context, self.beta = artifact, context, float(beta)
        self.control_names = tuple(name.removeprefix("control::") for name in artifact.input_schema.control_features)

    def _distribution(self, pool: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
        if set(pool.columns) - {"recipe_id", *self.control_names}: raise ValueError("candidate pool includes context, target, or metadata as an action")
        controls = pool.loc[:, self.control_names].to_dict(orient="records")
        return self.artifact.predict_decision(self.context, controls)[target]

    def score_candidates(self, observations: pd.DataFrame | Sequence[Mapping[str, Any]], candidate_pool: pd.DataFrame, objective: OptimizationObjective | str, *, candidate_id_column: str | None = None, **_: Any) -> dict[str, float]:
        target, minimize = (objective, False) if isinstance(objective, str) else (objective.target_name, objective.minimize)
        if target not in self.artifact.target_names: raise ValueError(f"target {target!r} is absent from frozen surrogate artifact")
        mean, std = self._distribution(candidate_pool, target); score = mean + self.beta * std if not minimize else -(mean - self.beta * std)
        return dict(zip(candidate_pool[candidate_id_column or "recipe_id"].astype(str), map(float, score)))

    def propose(self, observations: pd.DataFrame | Sequence[Mapping[str, Any]], candidate_pool: pd.DataFrame, objective: OptimizationObjective | str, *, candidate_id_column: str | None = None, n: int = 1, seed: int | None = None, **_: Any) -> list[CandidateProposal]:
        target, minimize = (objective, False) if isinstance(objective, str) else (objective.target_name, objective.minimize)
        id_col, ids = candidate_id_column or "recipe_id", candidate_pool[candidate_id_column or "recipe_id"].astype(str).tolist()
        mean, std = self._distribution(candidate_pool, target); scores = self.score_candidates(observations, candidate_pool, objective, candidate_id_column=id_col)
        return [CandidateProposal(candidate_id=ids[index], design_variables=candidate_pool.iloc[index][list(self.control_names)].to_dict(), predicted_mean=float(mean[index]), predicted_std=float(std[index]), acquisition_name="artifact_ucb", acquisition_value=scores[ids[index]], backend_name=self.name, backend_version=self.version, seed=seed, reason_code="FROZEN_ARTIFACT_FIXED_CONTEXT", recommendation_reason="fixed-context frozen-surrogate action ranking", metadata={"artifact_fingerprint": self.artifact.artifact_fingerprint, "context_fingerprint": self.context.context_fingerprint, "context_stage": self.context.stage.value, "objective_sense": "minimize" if minimize else "maximize"}) for index in sorted(range(len(ids)), key=lambda index: scores[ids[index]], reverse=True)[:n]]


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _samples(samples: Sequence[ProcessSurrogateSample], ids: Sequence[str]) -> list[ProcessSurrogateSample]:
    index = {sample.sample_id: sample for sample in samples}; return [index[item] for item in ids]


def _decision_context(sample: ProcessSurrogateSample) -> SurrogateDecisionContext:
    return SurrogateDecisionContext(sample.stage, sample.previous_state, sample.observations, sample.modality_state, sample.fidelity, sample.requested_horizon, {"source_run_id": sample.run_id, "source_dataset": sample.source_dataset, **sample.provenance})


def _offline_ranking(artifact: SurrogateArtifact, samples: Sequence[ProcessSurrogateSample], *, target: str, sense: str, top_k: int = 3) -> dict[str, Any]:
    if len(samples) < 2: return {"status": "NOT_AVAILABLE", "reason": "fewer than two held-out candidates", "mode": "OFFLINE_RANKING_EVALUATION"}
    predicted, actual = artifact.predict(samples)[target][0], np.asarray([sample.targets[target] for sample in samples], dtype=float)
    order = np.argsort(predicted if sense == "minimize" else -predicted); truth = np.argsort(actual if sense == "minimize" else -actual); selected, oracle, k = int(order[0]), int(truth[0]), min(top_k, len(samples))
    rank = float(pd.Series(predicted).corr(pd.Series(actual), method="spearman")); regret = actual[selected] - actual[oracle] if sense == "minimize" else actual[oracle] - actual[selected]
    return {"status": "AVAILABLE", "mode": "OFFLINE_RANKING_EVALUATION", "top1_match": bool(selected == oracle), "top_k_recall": float(len(set(order[:k]) & set(truth[:k])) / k), "spearman_rank_correlation": rank, "simple_regret": float(regret), "selected_sample_id": samples[selected].sample_id, "oracle_sample_id": samples[oracle].sample_id}


def _proposals(artifact: SurrogateArtifact, train: Sequence[ProcessSurrogateSample], candidates: Sequence[ProcessSurrogateSample], config: PipelineConfig) -> tuple[list[dict[str, Any]], SurrogateDecisionContext]:
    context = _decision_context(candidates[0]); names = tuple(name.removeprefix("control::") for name in artifact.input_schema.control_features)
    controls = pd.DataFrame([sample.controls for sample in candidates]).drop_duplicates().reset_index(drop=True)
    pool = pd.DataFrame({"recipe_id": [f"action-{index}" for index in range(len(controls))], **{name: controls[name] for name in names}})
    observed = pd.DataFrame({"recipe_id": [sample.sample_id for sample in train], config.objective_target or config.targets[0]: [sample.targets[config.objective_target or config.targets[0]] for sample in train]})
    coordinator = ProcessOptimizationCoordinator(scalar_backend=ArtifactOptimizerBackend(artifact, context, beta=config.exploration_beta)); target = config.objective_target or config.targets[0]
    results = coordinator.propose_recipes(observed, ProcessSearchSpace.from_finite_pool(pool), ProcessOptimizationObjective([ObjectiveSpec(target, config.objective_sense)]), n=min(config.proposal_count, len(pool)), seed=config.seed)
    return [{"proposal_id": item.proposal_id, "candidate_instance_id": item.candidate_instance_id, "source_recipe_id": item.source_recipe_id, "controls": item.controls, "predicted_outputs": {name: {"mean": output.mean, "std": output.std, "units": output.units} for name, output in item.predicted_outputs.items()}, "acquisition_value": item.acquisition_value, "model_version": item.model_version, "data_fingerprint": item.data_fingerprint, "provenance": item.provenance} for item in results], context


def _latency(artifact: SurrogateArtifact, context: SurrogateDecisionContext, controls: Sequence[Mapping[str, float]], target: str) -> dict[str, dict[str, float]]:
    measurements: dict[str, list[float]] = {"preprocessing_ms": [], "surrogate_prediction_ms": [], "total_decision_ms": []}
    rows = [{**context.feature_values(), **{f"control::{name}": value for name, value in control.items()}} for control in controls]
    for _ in range(20):
        start = time.perf_counter(); matrix = artifact.preprocessor.transform_values(rows); measurements["preprocessing_ms"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter(); artifact.surrogate.predict_distribution(matrix)[target]; measurements["surrogate_prediction_ms"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter(); artifact.predict_decision(context, controls)[target]; measurements["total_decision_ms"].append((time.perf_counter() - start) * 1000)
    return {name: {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99))} for name, values in measurements.items()}


def _queue_status(manifest_name: str) -> str:
    if manifest_name.startswith("synthetic"): return "SYNTHETIC_TEST_ONLY"
    if manifest_name == "artistic": return "READY_FOR_EXPLICIT_REVALIDATION"
    return "REQUIRES_SOURCE_BACKED_ARTISTIC_RECIPE_MAPPING"


def _git_commit() -> str:
    try: return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return "UNAVAILABLE"


def _reference_validation(samples: Sequence[ProcessSurrogateSample], validation_set: Sequence[ProcessSurrogateSample], test_set: Sequence[ProcessSurrogateSample]) -> dict[str, object]:
    present = any(sample.fidelity == "REFERENCE" for sample in samples)
    held_out = sum(sample.fidelity == "REFERENCE" for sample in (*validation_set, *test_set))
    return {"reference_fidelity_present": present, "status": "NOT_AVAILABLE" if not present else "EVALUATED" if held_out else "NOT_EVALUATED", "held_out_reference_sample_count": held_out}


def _experiment_id(dataset_fingerprint: str, split_fingerprint: str, config_fingerprint: str, surrogate_artifact_fingerprint: str) -> str:
    payload = [dataset_fingerprint, split_fingerprint, config_fingerprint, surrogate_artifact_fingerprint]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]


def _report_markdown(manifest: Any, artifact: SurrogateArtifact, metrics: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]], synthetic: bool) -> str:
    banner = "# SOFTWARE TEST ONLY — NOT SCIENTIFIC EVIDENCE\n\n" if synthetic else "# Dataset-driven surrogate experiment\n\n"
    reference = artifact.metadata()["reference_validation_status"]
    return banner + f"- Dataset: `{manifest.dataset_name}` (`{manifest.dataset_fingerprint}`)\n- Stage: `{artifact.input_schema.supported_stage}`\n- Model: `{artifact.surrogate.model_type}`; uncertainty: `{artifact.surrogate.uncertainty_kind}`\n- Supported fidelities: `{', '.join(artifact.input_schema.fidelity_vocabulary)}`\n- Reference fidelity present: `{artifact.metadata()['reference_fidelity_present']}`; held-out reference validation: `{reference}`.\n- Offline ranking: `{metrics['offline_ranking']}`\n- Fixed-context MASPO smoke proposals: `{len(proposals)}`\n- Claim boundary: no ARTISTIC simulator is invoked by this pipeline.\n"


def run_pipeline(adapter: DatasetAdapter, output_dir: str | Path, config: PipelineConfig) -> dict[str, Any]:
    """One source-audited software experiment; this function never launches ARTISTIC."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); manifest, validation, samples = adapter.manifest(), adapter.validate(targets=config.targets), adapter.samples()
    _json(output / "dataset_manifest.json", asdict(manifest)); _json(output / "validation_report.json", asdict(validation))
    if not validation.valid: raise ValueError("dataset validation failed: " + "; ".join(validation.errors))
    if len({sample.stage for sample in samples}) != 1: raise ValueError("current pipeline is explicitly stage-specific; mixed-stage training is rejected")
    split = split_groups(samples, manifest, seed=config.seed, validation_fraction=config.validation_fraction, test_fraction=config.test_fraction); train, validation_set, test = (_samples(samples, ids) for ids in (split.train_ids, split.validation_ids, split.test_ids))
    schema = SurrogateInputSchema.from_training_samples(train, declared_fidelities=manifest.fidelity_levels); preprocessor = TrainOnlyPreprocessor().fit(train, schema)
    surrogate = ProcessSurrogate(config.model_type, seed=config.seed).fit(preprocessor.transform(train), {target: np.asarray([sample.targets[target] for sample in train]) for target in config.targets})
    artifact = SurrogateArtifact(surrogate, preprocessor, manifest.dataset_fingerprint, split.split_fingerprint, config.targets, schema, dict(config.target_units or {target: manifest.units.get(target, "UNSPECIFIED") for target in config.targets}), training_config=asdict(config))
    validation_metrics, validation_predictions = evaluate(artifact, validation_set); test_metrics, test_predictions = evaluate(artifact, test); target = config.objective_target or config.targets[0]
    offline = _offline_ranking(artifact, test, target=target, sense=config.objective_sense); proposals, context = _proposals(artifact, train, test, config); controls = [proposal["controls"] for proposal in proposals]
    reference_validation = _reference_validation(samples, validation_set, test)
    artifact.metrics = {"validation": validation_metrics, "test": test_metrics, "offline_ranking": offline, "reference_validation": reference_validation, "residual_correction": "NOT_CONFIGURED"}; artifact_path = artifact.save(output / "surrogate.joblib")
    latency = _latency(artifact, context, controls, target); queue_status = _queue_status(manifest.dataset_name); queue = [{"proposal": proposal, "status": queue_status, "required_before_execution": ["explicit user approval", "published-domain ARTISTIC preflight"]} for proposal in proposals]
    pd.concat((validation_predictions.assign(split="validation"), test_predictions.assign(split="test"))).to_csv(output / "predictions.csv", index=False); _json(output / "split_manifest.json", asdict(split)); _json(output / "metrics.json", {"validation": validation_metrics, "test": test_metrics, "offline_ranking": offline, "reference_validation": reference_validation, "latency": latency}); _json(output / "proposals.json", proposals); _json(output / "revalidation_queue.json", queue)
    config_fingerprint = hashlib.sha256(json.dumps(asdict(config), sort_keys=True, default=str).encode()).hexdigest(); created = datetime.now(UTC).isoformat(); experiment_id = _experiment_id(manifest.dataset_fingerprint, split.split_fingerprint, config_fingerprint, artifact.artifact_fingerprint)
    report = {"status": "SUCCESS_SYNTHETIC_ONLY" if manifest.dataset_name.startswith("synthetic") else "SUCCESS_DATASET_BACKED", "experiment_id": experiment_id, "created_at": created, "git_commit": _git_commit(), "dataset_fingerprint": manifest.dataset_fingerprint, "split_fingerprint": split.split_fingerprint, "surrogate_artifact_fingerprint": artifact.artifact_fingerprint, "artifact_file_sha256": artifact.artifact_file_sha256, "config_fingerprint": config_fingerprint, "seed": config.seed, "supported_stage": schema.supported_stage, "supported_fidelities": schema.fidelity_vocabulary, "reference_fidelity_present": reference_validation["reference_fidelity_present"], "reference_validation_status": reference_validation["status"], "artifact": artifact.metadata(), "files": {"metrics": "metrics.json", "predictions": "predictions.csv", "proposals": "proposals.json", "revalidation_queue": "revalidation_queue.json", "report": "report.md"}, "claims": ["Synthetic E2E validates software plumbing only; it is not scientific performance evidence." if manifest.dataset_name.startswith("synthetic") else "Dataset evidence is bounded by its manifest.", "No ARTISTIC simulator was invoked by this pipeline."]}
    _json(output / "experiment_manifest.json", report); (output / "report.md").write_text(_report_markdown(manifest, artifact, {"offline_ranking": offline}, proposals, manifest.dataset_name.startswith("synthetic")), encoding="utf-8"); return report

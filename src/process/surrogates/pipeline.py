from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.backend import OptimizerBackend
from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace

from .core import DatasetAdapter, ProcessSurrogate, ProcessSurrogateSample, SurrogateArtifact, TrainOnlyPreprocessor, evaluate, split_groups


@dataclass(frozen=True)
class PipelineConfig:
    targets: tuple[str, ...]
    model_type: str = "gp"
    seed: int = 42
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    objective_target: str | None = None
    objective_sense: str = "maximize"
    proposal_count: int = 3
    exploration_beta: float = 1.0

    def __post_init__(self) -> None:
        if not self.targets or self.model_type not in {"gp", "extra_trees"} or self.objective_sense not in {"maximize", "minimize"}:
            raise ValueError("targets, model type, and objective sense are invalid")
        if self.objective_target is not None and self.objective_target not in self.targets:
            raise ValueError("objective target must be one of the fitted targets")


class ArtifactOptimizerBackend:
    """MASPO-compatible finite-pool scorer over a frozen, provenance-checked artifact."""

    name = "frozen_process_surrogate"
    version = "1"

    def __init__(self, artifact: SurrogateArtifact, *, beta: float = 1.0) -> None:
        self.artifact, self.beta = artifact, float(beta)

    def _matrix(self, candidates: pd.DataFrame) -> np.ndarray:
        names = self.artifact.preprocessor.feature_names
        raw = candidates.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
        values, observed = raw.fillna(0.0).to_numpy(dtype=float), raw.notna().to_numpy(dtype=float)
        return np.column_stack((self.artifact.preprocessor.scaler.transform(values), observed))

    def score_candidates(self, observations: pd.DataFrame | Sequence[Mapping[str, Any]], candidate_pool: pd.DataFrame, objective: OptimizationObjective | str, *, candidate_id_column: str | None = None, **_: Any) -> dict[str, float]:
        target, minimize = (objective, False) if isinstance(objective, str) else (objective.target_name, objective.minimize)
        if target not in self.artifact.target_names:
            raise ValueError(f"target {target!r} is absent from frozen surrogate artifact")
        mean, std = self.artifact.surrogate.predict_distribution(self._matrix(candidate_pool))[target]
        score = mean + self.beta * std if not minimize else -(mean - self.beta * std)
        ids = candidate_pool[candidate_id_column or "recipe_id"].astype(str)
        return dict(zip(ids, map(float, score)))

    def propose(self, observations: pd.DataFrame | Sequence[Mapping[str, Any]], candidate_pool: pd.DataFrame, objective: OptimizationObjective | str, *, candidate_id_column: str | None = None, n: int = 1, seed: int | None = None, **_: Any) -> list[CandidateProposal]:
        target, minimize = (objective, False) if isinstance(objective, str) else (objective.target_name, objective.minimize)
        ids = candidate_pool[candidate_id_column or "recipe_id"].astype(str).tolist()
        distribution = self.artifact.surrogate.predict_distribution(self._matrix(candidate_pool))[target]
        scores = self.score_candidates(observations, candidate_pool, objective, candidate_id_column=candidate_id_column)
        order = sorted(range(len(ids)), key=lambda index: scores[ids[index]], reverse=True)[:n]
        return [CandidateProposal(
            candidate_id=ids[index], design_variables=candidate_pool.iloc[index].drop(labels=[candidate_id_column or "recipe_id"], errors="ignore").dropna().to_dict(),
            predicted_mean=float(distribution[0][index]), predicted_std=float(distribution[1][index]), acquisition_name="artifact_ucb",
            acquisition_value=scores[ids[index]], backend_name=self.name, backend_version=self.version, seed=seed,
            reason_code="FROZEN_ARTIFACT_RANKING", recommendation_reason="ranked by frozen surrogate mean plus uncertainty", metadata={"artifact_fingerprint": self.artifact.artifact_fingerprint, "objective_sense": "minimize" if minimize else "maximize"},
        ) for index in order]


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _samples_by_id(samples: Sequence[ProcessSurrogateSample], ids: Sequence[str]) -> list[ProcessSurrogateSample]:
    index = {sample.sample_id: sample for sample in samples}
    return [index[item] for item in ids]


def _decision_metric(artifact: SurrogateArtifact, samples: Sequence[ProcessSurrogateSample], *, target: str, sense: str) -> dict[str, Any]:
    mean, _ = artifact.predict(samples)[target]
    actual = np.asarray([sample.targets[target] for sample in samples], dtype=float)
    choose = int(np.argmin(mean) if sense == "minimize" else np.argmax(mean))
    optimal = int(np.argmin(actual) if sense == "minimize" else np.argmax(actual))
    regret = actual[choose] - actual[optimal] if sense == "minimize" else actual[optimal] - actual[choose]
    return {"selected_sample_id": samples[choose].sample_id, "oracle_sample_id": samples[optimal].sample_id, "selected_actual": float(actual[choose]), "oracle_actual": float(actual[optimal]), "simple_regret": float(regret), "top1_match": bool(choose == optimal)}


def _proposals(artifact: SurrogateArtifact, train: Sequence[ProcessSurrogateSample], candidates: Sequence[ProcessSurrogateSample], config: PipelineConfig) -> list[dict[str, Any]]:
    pool = pd.DataFrame([{"recipe_id": sample.sample_id, **sample.feature_values()} for sample in candidates])
    observed = pd.DataFrame([{"recipe_id": sample.sample_id, **sample.feature_values(), **sample.targets} for sample in train])
    backend = ArtifactOptimizerBackend(artifact, beta=config.exploration_beta)
    coordinator = ProcessOptimizationCoordinator(scalar_backend=backend)
    objective_target = config.objective_target or config.targets[0]
    result = coordinator.propose_recipes(
        observed, ProcessSearchSpace.from_finite_pool(pool), ProcessOptimizationObjective([ObjectiveSpec(objective_target, config.objective_sense)]),
        n=min(config.proposal_count, len(pool)), seed=config.seed,
    )
    return [{
        "proposal_id": item.proposal_id, "candidate_instance_id": item.candidate_instance_id, "source_recipe_id": item.source_recipe_id,
        "controls": item.controls, "predicted_outputs": {name: {"mean": output.mean, "std": output.std, "units": output.units} for name, output in item.predicted_outputs.items()},
        "acquisition_value": item.acquisition_value, "model_version": item.model_version, "data_fingerprint": item.data_fingerprint, "provenance": item.provenance,
    } for item in result]


def run_pipeline(adapter: DatasetAdapter, output_dir: str | Path, config: PipelineConfig) -> dict[str, Any]:
    """Train/evaluate one small surrogate experiment; it never invokes a simulator."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest, validation, samples = adapter.manifest(), adapter.validate(targets=config.targets), adapter.samples()
    _json(output / "dataset_manifest.json", asdict(manifest)); _json(output / "validation_report.json", asdict(validation))
    if not validation.valid:
        raise ValueError("dataset validation failed: " + "; ".join(validation.errors))
    split = split_groups(samples, manifest, seed=config.seed, validation_fraction=config.validation_fraction, test_fraction=config.test_fraction)
    train, validation_set, test = (_samples_by_id(samples, ids) for ids in (split.train_ids, split.validation_ids, split.test_ids))
    preprocessor = TrainOnlyPreprocessor().fit(train)
    surrogate = ProcessSurrogate(config.model_type, seed=config.seed).fit(preprocessor.transform(train), {target: np.asarray([sample.targets[target] for sample in train]) for target in config.targets})
    artifact = SurrogateArtifact(surrogate, preprocessor, manifest.dataset_fingerprint, split.split_fingerprint, config.targets, training_config=asdict(config))
    validation_metrics, validation_predictions = evaluate(artifact, validation_set)
    test_metrics, test_predictions = evaluate(artifact, test)
    artifact.metrics = {"validation": validation_metrics, "test": test_metrics, "residual_correction": "NOT_CONFIGURED"}
    artifact_path = artifact.save(output / "surrogate.joblib")
    started = time.perf_counter(); artifact.predict(test * 20); latency_ms = (time.perf_counter() - started) * 1000 / max(1, len(test) * 20)
    objective_target = config.objective_target or config.targets[0]
    decision = _decision_metric(artifact, test, target=objective_target, sense=config.objective_sense)
    proposals = _proposals(artifact, train, test, config)
    queue = [{"proposal": proposal, "status": "SYNTHETIC_ONLY_NOT_ELIGIBLE_FOR_ARTISTIC_EXECUTION", "required_before_execution": ["source-backed ARTISTIC recipe mapping", "published-domain preflight", "explicit user approval"]} for proposal in proposals]
    pd.concat((validation_predictions.assign(split="validation"), test_predictions.assign(split="test"))).to_csv(output / "predictions.csv", index=False)
    _json(output / "split_manifest.json", asdict(split)); _json(output / "metrics.json", {"validation": validation_metrics, "test": test_metrics, "decision": decision, "latency_ms_per_prediction": latency_ms})
    _json(output / "proposals.json", proposals); _json(output / "revalidation_queue.json", queue)
    report = {
        "status": "SUCCESS_SYNTHETIC_ONLY" if manifest.dataset_name.startswith("synthetic") else "SUCCESS_DATASET_BACKED",
        "dataset_fingerprint": manifest.dataset_fingerprint, "split_fingerprint": split.split_fingerprint, "artifact": artifact.metadata(), "artifact_path": str(artifact_path),
        "metrics": {"validation": validation_metrics, "test": test_metrics, "decision": decision, "latency_ms_per_prediction": latency_ms},
        "claims": ["Synthetic E2E validates software plumbing only; it is not scientific performance evidence.", "No ARTISTIC simulator was invoked by this pipeline."],
    }
    _json(output / "experiment_manifest.json", report)
    return report

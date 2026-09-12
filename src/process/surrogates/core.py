from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.process.models.flat_baseline import GaussianProcessBaseline, TreeEnsembleBaseline
from src.process.stages import ProcessStage


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()).hexdigest()


def _init_values(value: object) -> dict[str, object]:
    return {item.name: getattr(value, item.name) for item in fields(value) if item.init}


@dataclass(frozen=True)
class ProcessSurrogateSample:
    sample_id: str
    run_id: str
    recipe_id: str | None
    source_dataset: str
    source_evidence_kind: str
    stage: ProcessStage
    group_id: str
    controls: Mapping[str, float]
    observations: Mapping[str, float] = field(default_factory=dict)
    previous_state: Mapping[str, float] = field(default_factory=dict)
    modality_state: Mapping[str, float] = field(default_factory=dict)
    fidelity: str = "UNKNOWN"
    requested_horizon: int | None = None
    physics_config_fingerprint: str | None = None
    targets: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    dataset_manifest_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in (self.sample_id, self.run_id, self.source_dataset, self.source_evidence_kind, self.group_id)):
            raise ValueError("sample_id, run_id, source dataset/evidence, and group_id are required")
        for name, values in (("controls", self.controls), ("observations", self.observations), ("previous_state", self.previous_state), ("modality_state", self.modality_state), ("targets", self.targets)):
            for key, value in values.items():
                if not str(key).strip() or not np.isfinite(float(value)):
                    raise ValueError(f"{name} contains a missing or non-finite numeric value")
        if self.requested_horizon is not None and self.requested_horizon < 1:
            raise ValueError("requested_horizon must be positive when supplied")

    def feature_values(self) -> dict[str, float]:
        values: dict[str, float] = {"fidelity::" + self.fidelity: 1.0}
        for prefix, source in (("control", self.controls), ("previous", self.previous_state), ("observation", self.observations), ("modality", self.modality_state)):
            values.update({f"{prefix}::{name}": float(value) for name, value in source.items()})
        if self.requested_horizon is not None:
            values["fidelity::requested_horizon"] = float(self.requested_horizon)
        return values


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    dataset_version: str
    source: str
    source_hashes: Mapping[str, str]
    adapter_name: str
    adapter_version: str
    chemistry: str | None
    process_stages: tuple[str, ...]
    available_controls: tuple[str, ...]
    available_observations: tuple[str, ...]
    available_targets: tuple[str, ...]
    available_modalities: tuple[str, ...]
    fidelity_levels: tuple[str, ...]
    sample_count: int
    grouping_key: str
    leakage_grouping_policy: str
    units: Mapping[str, str] = field(default_factory=dict)
    dataset_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.dataset_name.strip() or self.sample_count < 0 or not self.grouping_key.strip():
            raise ValueError("dataset manifest requires name, non-negative sample_count, and grouping key")
        object.__setattr__(self, "dataset_fingerprint", _fingerprint(_init_values(self)))


@dataclass(frozen=True)
class DatasetValidationReport:
    status: str
    sample_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    field_summaries: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    fidelity_distribution: Mapping[str, int] = field(default_factory=dict)
    stage_distribution: Mapping[str, int] = field(default_factory=dict)
    target_availability: Mapping[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "VALID"


@dataclass(frozen=True)
class SplitManifest:
    split_seed: int
    group_key: str
    source_dataset_fingerprint: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    split_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        all_ids = [*self.train_ids, *self.validation_ids, *self.test_ids]
        if len(all_ids) != len(set(all_ids)) or not all(self.train_ids) or not all(self.validation_ids) or not all(self.test_ids):
            raise ValueError("split IDs must be non-empty and mutually exclusive")
        object.__setattr__(self, "split_fingerprint", _fingerprint(_init_values(self)))


class DatasetAdapter(Protocol):
    def samples(self) -> list[ProcessSurrogateSample]: ...
    def manifest(self) -> DatasetManifest: ...
    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport: ...


class ResidualCorrection(Protocol):
    """Optional held-out residual correction; implementations must declare calibration evidence."""

    def predict(self, X: np.ndarray, *, target: str) -> tuple[np.ndarray, np.ndarray | None]: ...


class GenericTabularAdapter:
    """Thin config-driven adapter for one tabular evidence source; never a master table."""

    ADAPTER_VERSION = "1"

    def __init__(self, frame: pd.DataFrame, mapping: Mapping[str, Any], *, source: str = "tabular") -> None:
        self.frame, self.mapping, self.source = frame.copy(), dict(mapping), source

    def _field(self, name: str, *, required: bool = True) -> str | None:
        value = self.mapping.get(name)
        if required and (not isinstance(value, str) or value not in self.frame):
            raise ValueError(f"mapping requires existing column {name!r}")
        return value if isinstance(value, str) else None

    def _fields(self, name: str) -> list[str]:
        values = self.mapping.get(name, [])
        if not isinstance(values, list) or any(not isinstance(item, str) or item not in self.frame for item in values):
            raise ValueError(f"mapping {name!r} must contain existing columns")
        return values

    def samples(self) -> list[ProcessSurrogateSample]:
        sample_col, run_col, group_col, stage_col = (self._field(name) for name in ("sample_id", "run_id", "group_id", "stage"))
        recipe_col = self._field("recipe_id", required=False)
        fidelity_col = self._field("fidelity", required=False)
        horizon_col = self._field("requested_horizon", required=False)
        controls, observations, previous, modalities, targets = (self._fields(name) for name in ("controls", "observations", "previous_state", "modality_state", "targets"))
        manifest = self.manifest()
        result: list[ProcessSurrogateSample] = []
        for _, row in self.frame.iterrows():
            try:
                stage = ProcessStage(str(row[stage_col]))
                numeric = lambda names: {name: float(row[name]) for name in names if pd.notna(row[name])}
                result.append(ProcessSurrogateSample(
                    sample_id=str(row[sample_col]), run_id=str(row[run_col]), recipe_id=str(row[recipe_col]) if recipe_col and pd.notna(row[recipe_col]) else None,
                    source_dataset=manifest.dataset_name, source_evidence_kind=str(self.mapping.get("evidence_kind", "EXTERNAL_TABULAR")),
                    stage=stage, group_id=str(row[group_col]), controls=numeric(controls), observations=numeric(observations), previous_state=numeric(previous), modality_state=numeric(modalities),
                    fidelity=str(row[fidelity_col]) if fidelity_col and pd.notna(row[fidelity_col]) else "UNKNOWN",
                    requested_horizon=int(row[horizon_col]) if horizon_col and pd.notna(row[horizon_col]) else None,
                    targets=numeric(targets), provenance={"source_row": int(row.name)}, dataset_manifest_fingerprint=manifest.dataset_fingerprint,
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid row {row.name}: {exc}") from exc
        return result

    def manifest(self) -> DatasetManifest:
        controls, observations, targets = (self._fields(name) for name in ("controls", "observations", "targets"))
        return DatasetManifest(
            dataset_name=str(self.mapping.get("dataset_name", "generic-tabular")), dataset_version=str(self.mapping.get("dataset_version", "unspecified")), source=self.source,
            source_hashes={"frame": _fingerprint(self.frame.to_dict(orient="records"))}, adapter_name=type(self).__name__, adapter_version=self.ADAPTER_VERSION,
            chemistry=self.mapping.get("chemistry"), process_stages=tuple(sorted({str(value) for value in self.frame[self._field("stage")].dropna()})),
            available_controls=tuple(controls), available_observations=tuple(observations), available_targets=tuple(targets), available_modalities=tuple(self._fields("modality_state")),
            fidelity_levels=tuple(sorted({str(value) for value in self.frame[self._field("fidelity", required=False)].dropna()})) if self._field("fidelity", required=False) else (),
            sample_count=len(self.frame), grouping_key=str(self._field("group_id")), leakage_grouping_policy="GROUP_ISOLATED", units=dict(self.mapping.get("units", {})),
        )

    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport:
        errors: list[str] = []
        try:
            samples = self.samples()
        except ValueError as exc:
            return DatasetValidationReport("INVALID", 0, (str(exc),))
        ids = [sample.sample_id for sample in samples]
        if len(ids) != len(set(ids)):
            errors.append("duplicate sample IDs")
        configured_targets = tuple(targets or self._fields("targets"))
        availability = {target: sum(target in sample.targets for sample in samples) for target in configured_targets}
        errors.extend(f"missing configured target: {target}" for target, count in availability.items() if count != len(samples))
        feature_fields = set(self._fields("controls") + self._fields("observations") + self._fields("previous_state") + self._fields("modality_state"))
        errors.extend(f"target is incorrectly exposed as a feature: {target}" for target in configured_targets if target in feature_fields)
        if len({sample.group_id for sample in samples}) < 3:
            errors.append("at least three leakage groups are required for train/validation/test")
        return DatasetValidationReport(
            "VALID" if not errors else "INVALID", len(samples), tuple(errors),
            field_summaries={"controls": {"count": len(self._fields("controls"))}, "targets": {"count": len(configured_targets)}},
            fidelity_distribution=dict(pd.Series([sample.fidelity for sample in samples]).value_counts()),
            stage_distribution=dict(pd.Series([sample.stage.value for sample in samples]).value_counts()), target_availability=availability,
        )


class ArtisticRunDirectoryAdapter:
    """Read-only bridge from an immutable normalized ARTISTIC cache to surrogate samples."""

    ADAPTER_VERSION = "1"

    def __init__(self, root: str | Path, *, stage: ProcessStage = ProcessStage.CALENDERING, targets: Sequence[str] = ()) -> None:
        self.root, self.stage, self.targets = Path(root), stage, tuple(targets)

    def _runs(self):
        from src.datasets.battery_process.artistic import ArtisticSimulationAdapter
        return ArtisticSimulationAdapter(self.root).load_runs()

    def manifest(self) -> DatasetManifest:
        aggregate = self.root / "manifest.json"
        if not aggregate.is_file():
            raise ValueError("ARTISTIC normalized cache manifest is required")
        raw = aggregate.read_bytes()
        runs = self._runs()
        return DatasetManifest(
            dataset_name="artistic", dataset_version="normalized-cache-v3", source=str(self.root),
            source_hashes={"manifest.json": hashlib.sha256(raw).hexdigest()}, adapter_name=type(self).__name__, adapter_version=self.ADAPTER_VERSION,
            chemistry="ARTISTIC_NMC", process_stages=(self.stage.value,), available_controls=(), available_observations=(),
            available_targets=self.targets, available_modalities=(), fidelity_levels=tuple(sorted({str(run.provenance.processing_parameters.get("fidelity_mode", "UNKNOWN")) for run in runs})),
            sample_count=len(runs), grouping_key="recipe_fingerprint", leakage_grouping_policy="RECIPE_FINGERPRINT_ISOLATED",
        )

    def samples(self) -> list[ProcessSurrogateSample]:
        manifest = self.manifest()
        result: list[ProcessSurrogateSample] = []
        for run in self._runs():
            parameters = run.provenance.processing_parameters
            recipe = str(parameters.get("recipe_fingerprint") or run.batch_id or "")
            if not recipe or not parameters.get("simulation_manifest_sha256") or not parameters.get("physics_config_fingerprint"):
                raise ValueError(f"ARTISTIC run {run.run_id} lacks immutable recipe/physics provenance")
            selected = next((item for item in run.stages if item.stage_type == self.stage), None)
            if selected is None:
                raise ValueError(f"ARTISTIC run {run.run_id} lacks requested stage {self.stage.value}")
            targets = {name: float(run.final_kpis[name].value) for name in self.targets if name in run.final_kpis}
            if len(targets) != len(self.targets):
                raise ValueError(f"ARTISTIC run {run.run_id} lacks configured final target")
            previous = {
                name: float(value.value) for item in run.stages if item.sequence_index < selected.sequence_index
                for name, value in item.intermediate_properties.items() if isinstance(value.value, (int, float))
            }
            result.append(ProcessSurrogateSample(
                sample_id=run.run_id, run_id=run.run_id, recipe_id=recipe, source_dataset=manifest.dataset_name,
                source_evidence_kind=run.provenance.evidence_kind, stage=self.stage, group_id=recipe,
                controls={name: float(value.value) for name, value in selected.controls.items() if isinstance(value.value, (int, float))},
                observations={name: float(value.value) for name, value in selected.intermediate_properties.items() if isinstance(value.value, (int, float))},
                previous_state=previous, fidelity=str(parameters.get("fidelity_mode", "UNKNOWN")),
                requested_horizon=int(parameters["requested_slurry_steps"]) if parameters.get("requested_slurry_steps") else None,
                physics_config_fingerprint=str(parameters["physics_config_fingerprint"]), targets=targets,
                provenance={"simulation_manifest": parameters.get("normalized_simulation_manifest"), "simulation_manifest_sha256": parameters["simulation_manifest_sha256"], "source_commit": parameters.get("source_commit"), "source_tree_hash": parameters.get("source_tree_hash"), "executable_identity": parameters.get("executable_identity", {})},
                dataset_manifest_fingerprint=manifest.dataset_fingerprint,
            ))
        return result

    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport:
        if targets is not None and tuple(targets) != self.targets:
            return DatasetValidationReport("INVALID", 0, ("configured ARTISTIC targets differ from adapter targets",))
        try:
            samples = self.samples()
        except (OSError, ValueError) as exc:
            return DatasetValidationReport("INVALID", 0, (str(exc),))
        errors = ["at least three recipe groups are required for train/validation/test"] if len({sample.group_id for sample in samples}) < 3 else []
        return DatasetValidationReport("VALID" if not errors else "INVALID", len(samples), tuple(errors), target_availability={target: len(samples) for target in self.targets})


def split_groups(samples: Sequence[ProcessSurrogateSample], manifest: DatasetManifest, *, seed: int, validation_fraction: float = 0.2, test_fraction: float = 0.2) -> SplitManifest:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id, []).append(sample.sample_id)
    if len(groups) < 3:
        raise ValueError("at least three groups are required for leakage-safe splitting")
    ordered = np.array(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed); rng.shuffle(ordered)
    test_n = max(1, round(len(ordered) * test_fraction)); validation_n = max(1, round(len(ordered) * validation_fraction))
    if len(ordered) - test_n - validation_n < 1:
        validation_n = 1; test_n = 1
    test_groups, validation_groups, train_groups = ordered[:test_n], ordered[test_n:test_n + validation_n], ordered[test_n + validation_n:]
    ids = lambda chosen: tuple(sorted(sample_id for group in chosen for sample_id in groups[str(group)]))
    return SplitManifest(seed, manifest.grouping_key, manifest.dataset_fingerprint, ids(train_groups), ids(validation_groups), ids(test_groups))


class TrainOnlyPreprocessor:
    def fit(self, samples: Sequence[ProcessSurrogateSample]) -> "TrainOnlyPreprocessor":
        self.feature_names = tuple(sorted({name for sample in samples for name in sample.feature_values()}))
        if not self.feature_names:
            raise ValueError("no features available for surrogate training")
        raw, masks = self._raw(samples)
        self.scaler = StandardScaler().fit(raw)
        self.output_names = tuple([*self.feature_names, *[f"{name}__observed" for name in self.feature_names]])
        return self

    def transform(self, samples: Sequence[ProcessSurrogateSample]) -> np.ndarray:
        if not hasattr(self, "scaler"):
            raise RuntimeError("preprocessor must be fit on training samples before transform")
        raw, masks = self._raw(samples)
        return np.column_stack((self.scaler.transform(raw), masks))

    def _raw(self, samples: Sequence[ProcessSurrogateSample]) -> tuple[np.ndarray, np.ndarray]:
        raw = np.zeros((len(samples), len(self.feature_names)), dtype=float); masks = np.zeros_like(raw)
        for row, sample in enumerate(samples):
            values = sample.feature_values()
            for col, name in enumerate(self.feature_names):
                if name in values:
                    raw[row, col], masks[row, col] = values[name], 1.0
        return raw, masks


class ProcessSurrogate:
    """Small common multi-target API: GP has posterior std; tree std is ensemble spread."""

    def __init__(self, model_type: str = "gp", *, seed: int = 42) -> None:
        if model_type not in {"gp", "extra_trees"}:
            raise ValueError("model_type must be gp or extra_trees")
        self.model_type, self.seed = model_type, seed

    def fit(self, X: np.ndarray, targets: Mapping[str, np.ndarray]) -> "ProcessSurrogate":
        self.models: dict[str, object] = {}
        for target, values in targets.items():
            model = GaussianProcessBaseline(random_state=self.seed) if self.model_type == "gp" else TreeEnsembleBaseline("extra_trees", random_state=self.seed)
            self.models[target] = model.fit(X, np.asarray(values, dtype=float))
        if not self.models:
            raise ValueError("at least one target is required")
        return self

    def predict_distribution(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        if not hasattr(self, "models"):
            raise RuntimeError("surrogate must be fit before prediction")
        return {target: model.predict_distribution(X) for target, model in self.models.items()}


@dataclass
class SurrogateArtifact:
    surrogate: ProcessSurrogate
    preprocessor: TrainOnlyPreprocessor
    dataset_fingerprint: str
    split_fingerprint: str
    target_names: tuple[str, ...]
    model_version: str = "process-surrogate-v1"
    training_config: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_fingerprint = _fingerprint({"model_type": self.surrogate.model_type, "dataset": self.dataset_fingerprint, "split": self.split_fingerprint, "features": self.preprocessor.output_names, "targets": self.target_names, "config": dict(self.training_config)})

    def predict(self, samples: Sequence[ProcessSurrogateSample]) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        return self.surrogate.predict_distribution(self.preprocessor.transform(samples))

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(self.metadata(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, *, expected_dataset_fingerprint: str | None = None, expected_features: Sequence[str] | None = None) -> "SurrogateArtifact":
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError("artifact is not a Process SurrogateArtifact")
        if expected_dataset_fingerprint and artifact.dataset_fingerprint != expected_dataset_fingerprint:
            raise ValueError("surrogate artifact dataset fingerprint mismatch")
        if expected_features and tuple(expected_features) != artifact.preprocessor.output_names:
            raise ValueError("surrogate artifact input schema mismatch")
        return artifact

    def metadata(self) -> dict[str, Any]:
        return {"model_type": self.surrogate.model_type, "model_version": self.model_version, "dataset_fingerprint": self.dataset_fingerprint, "split_fingerprint": self.split_fingerprint, "target_schema": self.target_names, "input_schema": self.preprocessor.output_names, "preprocessing_fingerprint": _fingerprint(self.preprocessor.output_names), "training_config_fingerprint": _fingerprint(dict(self.training_config)), "metrics": self.metrics, "artifact_fingerprint": self.artifact_fingerprint}


def evaluate(artifact: SurrogateArtifact, samples: Sequence[ProcessSurrogateSample]) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted = artifact.predict(samples); rows: list[dict[str, object]] = []; metrics: dict[str, Any] = {}
    for target, (mean, std) in predicted.items():
        actual = np.asarray([sample.targets[target] for sample in samples], dtype=float)
        rows.extend({"sample_id": sample.sample_id, "target": target, "actual": float(y), "predicted": float(p), "std": float(s) if std is not None else None} for sample, y, p, s in zip(samples, actual, mean, std if std is not None else np.repeat(np.nan, len(mean))))
        item: dict[str, Any] = {"mae": float(mean_absolute_error(actual, mean)), "rmse": float(mean_squared_error(actual, mean) ** 0.5), "r2": float(r2_score(actual, mean)) if len(actual) > 1 and np.var(actual) else None}
        if std is not None:
            safe = np.maximum(std, 1e-8); item["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * safe**2) + (actual - mean) ** 2 / (2 * safe**2)))
            item["coverage"] = {str(level): float(np.mean(np.abs(actual - mean) <= z * safe)) for level, z in ((50, 0.67448975), (80, 1.28155157), (95, 1.95996398))}
        else:
            item["uncertainty"] = "UNAVAILABLE"
        metrics[target] = item
    return metrics, pd.DataFrame(rows)

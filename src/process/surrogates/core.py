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

CONTRACT_VERSION = "2"


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()).hexdigest()


def _init_values(value: object) -> dict[str, object]:
    return {item.name: getattr(value, item.name) for item in fields(value) if item.init}


class _StateHasher:
    def __init__(self, tag: str) -> None:
        self.hash = hashlib.sha256(tag.encode("utf-8"))

    def value(self, name: str, value: object) -> None:
        self.hash.update(name.encode()); self.hash.update(b"\0")
        self.hash.update(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()); self.hash.update(b"\0")

    def array(self, name: str, value: object) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        self.value(name + ".header", {"dtype": array.dtype.str, "shape": array.shape})
        self.hash.update(array.tobytes(order="C")); self.hash.update(b"\0")

    def digest(self) -> str:
        return self.hash.hexdigest()


def _scaler_state(hasher: _StateHasher, scaler: StandardScaler, prefix: str) -> None:
    for name in ("mean_", "scale_", "var_"):
        if not hasattr(scaler, name): raise RuntimeError("unfitted scaler cannot be fingerprinted")
        hasher.array(prefix + name, getattr(scaler, name))
    hasher.value(prefix + "n_features_in_", int(scaler.n_features_in_))


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
        if not all(str(value).strip() for value in (self.sample_id, self.run_id, self.source_dataset, self.source_evidence_kind, self.group_id, self.fidelity)):
            raise ValueError("sample_id, run_id, source dataset/evidence, group_id, and fidelity are required")
        for name, values in (("controls", self.controls), ("observations", self.observations), ("previous_state", self.previous_state), ("modality_state", self.modality_state), ("targets", self.targets)):
            for key, value in values.items():
                if not str(key).strip() or not np.isfinite(float(value)): raise ValueError(f"{name} contains a missing or non-finite numeric value")
        if self.requested_horizon is not None and self.requested_horizon < 1: raise ValueError("requested_horizon must be positive when supplied")

    def feature_values(self) -> dict[str, float]:
        values: dict[str, float] = {"fidelity::" + self.fidelity: 1.0}
        for prefix, source in (("control", self.controls), ("previous", self.previous_state), ("observation", self.observations), ("modality", self.modality_state)):
            values.update({f"{prefix}::{name}": float(value) for name, value in source.items()})
        if self.requested_horizon is not None: values["fidelity::requested_horizon"] = float(self.requested_horizon)
        return values


@dataclass(frozen=True)
class SurrogateInputSchema:
    supported_stage: str
    control_features: tuple[str, ...]
    context_features: tuple[str, ...]
    feature_order: tuple[str, ...]
    stage_vocabulary: tuple[str, ...]
    fidelity_vocabulary: tuple[str, ...]
    schema_version: str = CONTRACT_VERSION
    schema_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.stage_vocabulary != (self.supported_stage,) or not self.supported_stage: raise ValueError("current surrogate artifacts must be explicitly stage-specific")
        if set(self.control_features) & set(self.context_features) or tuple([*self.control_features, *self.context_features]) != self.feature_order: raise ValueError("input schema requires disjoint ordered control and context features")
        if any(not name.startswith("control::") for name in self.control_features) or any(name.startswith("control::") for name in self.context_features): raise ValueError("input schema feature roles are invalid")
        object.__setattr__(self, "schema_fingerprint", _fingerprint(_init_values(self)))

    @classmethod
    def from_training_samples(cls, samples: Sequence[ProcessSurrogateSample], *, declared_fidelities: Sequence[str]) -> "SurrogateInputSchema":
        stages = {sample.stage.value for sample in samples}
        if len(stages) != 1: raise ValueError("stage-specific surrogate training requires exactly one process stage")
        observed = {sample.fidelity for sample in samples}; unknown = observed - set(declared_fidelities)
        if unknown: raise ValueError(f"training fidelity is absent from frozen dataset vocabulary: {sorted(unknown)}")
        values = {name for sample in samples for name in sample.feature_values()}; controls = tuple(sorted(name for name in values if name.startswith("control::"))); context = tuple(sorted(values - set(controls)))
        stage = next(iter(stages)); return cls(stage, controls, context, (*controls, *context), (stage,), tuple(sorted(observed)))


@dataclass(frozen=True)
class SurrogateDecisionContext:
    stage: ProcessStage
    previous_state: Mapping[str, float] = field(default_factory=dict)
    observations: Mapping[str, float] = field(default_factory=dict)
    modality_state: Mapping[str, float] = field(default_factory=dict)
    fidelity: str = "UNKNOWN"
    requested_horizon: int | None = None
    source_state_provenance: Mapping[str, Any] = field(default_factory=dict)
    context_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for values in (self.previous_state, self.observations, self.modality_state):
            if any(not np.isfinite(float(value)) for value in values.values()): raise ValueError("decision context contains non-finite state")
        object.__setattr__(self, "context_fingerprint", _fingerprint(_init_values(self)))

    def feature_values(self) -> dict[str, float]:
        values = {"fidelity::" + self.fidelity: 1.0}
        for prefix, source in (("previous", self.previous_state), ("observation", self.observations), ("modality", self.modality_state)):
            values.update({f"{prefix}::{name}": float(value) for name, value in source.items()})
        if self.requested_horizon is not None: values["fidelity::requested_horizon"] = float(self.requested_horizon)
        return values


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str; dataset_version: str; source: str; source_hashes: Mapping[str, str]; adapter_name: str; adapter_version: str; chemistry: str | None
    process_stages: tuple[str, ...]; available_controls: tuple[str, ...]; available_observations: tuple[str, ...]; available_targets: tuple[str, ...]; available_modalities: tuple[str, ...]; fidelity_levels: tuple[str, ...]
    sample_count: int; grouping_key: str; leakage_grouping_policy: str; units: Mapping[str, str] = field(default_factory=dict); dataset_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.dataset_name.strip() or self.sample_count < 0 or not self.grouping_key.strip() or not self.fidelity_levels: raise ValueError("dataset manifest requires name, non-negative sample_count, grouping key, and fidelity vocabulary")
        object.__setattr__(self, "dataset_fingerprint", _fingerprint(_init_values(self)))


@dataclass(frozen=True)
class DatasetValidationReport:
    status: str; sample_count: int; errors: tuple[str, ...] = (); warnings: tuple[str, ...] = (); field_summaries: Mapping[str, Mapping[str, object]] = field(default_factory=dict); fidelity_distribution: Mapping[str, int] = field(default_factory=dict); stage_distribution: Mapping[str, int] = field(default_factory=dict); target_availability: Mapping[str, int] = field(default_factory=dict)
    @property
    def valid(self) -> bool: return self.status == "VALID"


@dataclass(frozen=True)
class SplitManifest:
    split_seed: int; group_key: str; source_dataset_fingerprint: str; train_ids: tuple[str, ...]; validation_ids: tuple[str, ...]; test_ids: tuple[str, ...]; split_fingerprint: str = field(init=False)
    def __post_init__(self) -> None:
        values = [*self.train_ids, *self.validation_ids, *self.test_ids]
        if len(values) != len(set(values)) or not all(self.train_ids) or not all(self.validation_ids) or not all(self.test_ids): raise ValueError("split IDs must be non-empty and mutually exclusive")
        object.__setattr__(self, "split_fingerprint", _fingerprint(_init_values(self)))


class DatasetAdapter(Protocol):
    def samples(self) -> list[ProcessSurrogateSample]: ...
    def manifest(self) -> DatasetManifest: ...
    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport: ...


class ResidualCorrection(Protocol):
    def predict(self, X: np.ndarray, *, target: str) -> tuple[np.ndarray, np.ndarray | None]: ...


class GenericTabularAdapter:
    ADAPTER_VERSION = "2"
    def __init__(self, frame: pd.DataFrame, mapping: Mapping[str, Any], *, source: str = "tabular") -> None: self.frame, self.mapping, self.source = frame.copy(), dict(mapping), source
    def _field(self, name: str, *, required: bool = True) -> str | None:
        value = self.mapping.get(name)
        if required and (not isinstance(value, str) or value not in self.frame): raise ValueError(f"mapping requires existing column {name!r}")
        return value if isinstance(value, str) else None
    def _fields(self, name: str) -> list[str]:
        values = self.mapping.get(name, [])
        if not isinstance(values, list) or any(not isinstance(item, str) or item not in self.frame for item in values): raise ValueError(f"mapping {name!r} must contain existing columns")
        return values
    def manifest(self) -> DatasetManifest:
        controls, observations, targets = (self._fields(name) for name in ("controls", "observations", "targets")); fidelity = self._field("fidelity", required=False)
        return DatasetManifest(str(self.mapping.get("dataset_name", "generic-tabular")), str(self.mapping.get("dataset_version", "unspecified")), self.source, {"frame": _fingerprint(self.frame.to_dict(orient="records"))}, type(self).__name__, self.ADAPTER_VERSION, self.mapping.get("chemistry"), tuple(sorted({str(value) for value in self.frame[self._field("stage")].dropna()})), tuple(controls), tuple(observations), tuple(targets), tuple(self._fields("modality_state")), tuple(sorted({str(value) for value in self.frame[fidelity].dropna()})) if fidelity else ("UNKNOWN",), len(self.frame), str(self._field("group_id")), "GROUP_ISOLATED", dict(self.mapping.get("units", {})))
    def samples(self) -> list[ProcessSurrogateSample]:
        sample_col, run_col, group_col, stage_col = (self._field(name) for name in ("sample_id", "run_id", "group_id", "stage")); recipe, fidelity, horizon = (self._field(name, required=False) for name in ("recipe_id", "fidelity", "requested_horizon")); controls, observations, previous, modalities, targets = (self._fields(name) for name in ("controls", "observations", "previous_state", "modality_state", "targets")); manifest, result = self.manifest(), []
        for _, row in self.frame.iterrows():
            try:
                numeric = lambda names: {name: float(row[name]) for name in names if pd.notna(row[name])}
                result.append(ProcessSurrogateSample(str(row[sample_col]), str(row[run_col]), str(row[recipe]) if recipe and pd.notna(row[recipe]) else None, manifest.dataset_name, str(self.mapping.get("evidence_kind", "EXTERNAL_TABULAR")), ProcessStage(str(row[stage_col])), str(row[group_col]), numeric(controls), numeric(observations), numeric(previous), numeric(modalities), str(row[fidelity]) if fidelity and pd.notna(row[fidelity]) else "UNKNOWN", int(row[horizon]) if horizon and pd.notna(row[horizon]) else None, None, numeric(targets), {"source_row": int(row.name)}, manifest.dataset_fingerprint))
            except (TypeError, ValueError) as exc: raise ValueError(f"invalid row {row.name}: {exc}") from exc
        return result
    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport:
        try: samples, manifest = self.samples(), self.manifest()
        except ValueError as exc: return DatasetValidationReport("INVALID", 0, (str(exc),))
        errors = ["duplicate sample IDs"] if len({s.sample_id for s in samples}) != len(samples) else []; configured = tuple(targets or self._fields("targets")); availability = {target: sum(target in sample.targets for sample in samples) for target in configured}; errors.extend(f"missing configured target: {target}" for target, count in availability.items() if count != len(samples)); feature_fields = set(self._fields("controls") + self._fields("observations") + self._fields("previous_state") + self._fields("modality_state")); errors.extend(f"target is incorrectly exposed as a feature: {target}" for target in configured if target in feature_fields)
        if len({sample.group_id for sample in samples}) < 3: errors.append("at least three leakage groups are required for train/validation/test")
        schema = self.mapping.get("unit_schema", {})
        if schema and (not isinstance(schema, Mapping) or any(manifest.units.get(name) not in set(allowed if isinstance(allowed, list) else [allowed]) for name, allowed in schema.items())): errors.append("invalid or unknown unit schema")
        return DatasetValidationReport("VALID" if not errors else "INVALID", len(samples), tuple(errors), field_summaries={"controls": {"count": len(self._fields("controls"))}, "targets": {"count": len(configured)}}, fidelity_distribution=dict(pd.Series([s.fidelity for s in samples]).value_counts()), stage_distribution=dict(pd.Series([s.stage.value for s in samples]).value_counts()), target_availability=availability)


class ArtisticRunDirectoryAdapter:
    ADAPTER_VERSION = "2"
    def __init__(self, root: str | Path, *, stage: ProcessStage = ProcessStage.CALENDERING, targets: Sequence[str] = ()) -> None: self.root, self.stage, self.targets = Path(root), stage, tuple(targets)
    def _runs(self):
        from src.datasets.battery_process.artistic import ArtisticSimulationAdapter
        return ArtisticSimulationAdapter(self.root).load_runs()
    def manifest(self) -> DatasetManifest:
        path = self.root / "manifest.json"
        if not path.is_file(): raise ValueError("ARTISTIC normalized cache manifest is required")
        runs = self._runs(); fids = {str(run.provenance.processing_parameters.get("fidelity_mode", "UNKNOWN")) for run in runs}
        return DatasetManifest("artistic", "normalized-cache-v3", str(self.root), {"manifest.json": hashlib.sha256(path.read_bytes()).hexdigest()}, type(self).__name__, self.ADAPTER_VERSION, "ARTISTIC_NMC", (self.stage.value,), (), (), self.targets, (), tuple(sorted(fids or {"UNKNOWN"})), len(runs), "recipe_fingerprint", "RECIPE_FINGERPRINT_ISOLATED")
    def samples(self) -> list[ProcessSurrogateSample]:
        manifest, result = self.manifest(), []
        for run in self._runs():
            p, recipe, fidelity, evidence = run.provenance.processing_parameters, str(run.provenance.processing_parameters.get("recipe_fingerprint") or run.batch_id or ""), str(run.provenance.processing_parameters.get("fidelity_mode", "UNKNOWN")), run.provenance.evidence_kind
            if not recipe or not p.get("simulation_manifest_sha256") or not p.get("physics_config_fingerprint"): raise ValueError(f"ARTISTIC run {run.run_id} lacks immutable recipe/physics provenance")
            if evidence not in {"SIMULATED_PHYSICS", "SIMULATED_STRESS"} or (fidelity == "REFERENCE") != (evidence == "SIMULATED_PHYSICS"): raise ValueError(f"ARTISTIC run {run.run_id} has invalid evidence/fidelity semantics")
            selected = next((item for item in run.stages if item.stage_type == self.stage), None)
            if selected is None: raise ValueError(f"ARTISTIC run {run.run_id} lacks requested stage {self.stage.value}")
            final = {name: float(run.final_kpis[name].value) for name in self.targets if name in run.final_kpis}
            if len(final) != len(self.targets): raise ValueError(f"ARTISTIC run {run.run_id} lacks configured final target")
            prior = {name: float(value.value) for item in run.stages if item.sequence_index < selected.sequence_index for name, value in item.intermediate_properties.items() if isinstance(value.value, (int, float))}
            result.append(ProcessSurrogateSample(run.run_id, run.run_id, recipe, manifest.dataset_name, evidence, self.stage, recipe, {name: float(value.value) for name, value in selected.controls.items() if isinstance(value.value, (int, float))}, {name: float(value.value) for name, value in selected.intermediate_properties.items() if isinstance(value.value, (int, float))}, prior, {}, fidelity, int(p["requested_slurry_steps"]) if p.get("requested_slurry_steps") else None, str(p["physics_config_fingerprint"]), final, {"simulation_manifest": p.get("normalized_simulation_manifest"), "simulation_manifest_sha256": p["simulation_manifest_sha256"], "source_commit": p.get("source_commit"), "source_tree_hash": p.get("source_tree_hash")}, manifest.dataset_fingerprint))
        return result
    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport:
        if targets is not None and tuple(targets) != self.targets: return DatasetValidationReport("INVALID", 0, ("configured ARTISTIC targets differ from adapter targets",))
        try: samples = self.samples()
        except (OSError, ValueError) as exc: return DatasetValidationReport("INVALID", 0, (str(exc),))
        errors = ["at least three recipe groups are required for train/validation/test"] if len({s.group_id for s in samples}) < 3 else []
        return DatasetValidationReport("VALID" if not errors else "INVALID", len(samples), tuple(errors), target_availability={target: len(samples) for target in self.targets})


def split_groups(samples: Sequence[ProcessSurrogateSample], manifest: DatasetManifest, *, seed: int, validation_fraction: float = .2, test_fraction: float = .2) -> SplitManifest:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1: raise ValueError("split fractions must be positive and sum to less than one")
    groups: dict[str, list[str]] = {}
    for sample in samples: groups.setdefault(sample.group_id, []).append(sample.sample_id)
    if len(groups) < 3: raise ValueError("at least three groups are required for leakage-safe splitting")
    ordered = np.array(sorted(groups), dtype=object); np.random.default_rng(seed).shuffle(ordered); test_n, val_n = max(1, round(len(ordered) * test_fraction)), max(1, round(len(ordered) * validation_fraction))
    if len(ordered) - test_n - val_n < 1: test_n = val_n = 1
    ids = lambda selected: tuple(sorted(sample_id for group in selected for sample_id in groups[str(group)])); return SplitManifest(seed, manifest.grouping_key, manifest.dataset_fingerprint, ids(ordered[test_n + val_n:]), ids(ordered[test_n:test_n + val_n]), ids(ordered[:test_n]))


class TrainOnlyPreprocessor:
    CONTRACT_VERSION = CONTRACT_VERSION
    def fit(self, samples: Sequence[ProcessSurrogateSample], schema: SurrogateInputSchema) -> "TrainOnlyPreprocessor":
        self.schema, self.feature_names = schema, schema.feature_order
        if not self.feature_names: raise ValueError("no features available for surrogate training")
        raw, _ = self._raw_samples(samples); self.scaler = StandardScaler().fit(raw); self.output_names = (*self.feature_names, *(f"{name}__observed" for name in self.feature_names)); self.preprocessor_state_fingerprint = self.state_fingerprint(); return self
    def _raw_values(self, value_rows: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
        raw, masks, known = np.zeros((len(value_rows), len(self.feature_names))), np.zeros((len(value_rows), len(self.feature_names))), set(self.feature_names)
        for row, values in enumerate(value_rows):
            unknown = set(values) - known
            if unknown: raise ValueError(f"unknown structural feature(s): {sorted(unknown)}")
            for col, name in enumerate(self.feature_names):
                if name in values: raw[row, col], masks[row, col] = float(values[name]), 1.
        return raw, masks
    def _raw_samples(self, samples: Sequence[ProcessSurrogateSample]) -> tuple[np.ndarray, np.ndarray]:
        if any(s.stage.value != self.schema.supported_stage for s in samples): raise ValueError("sample stage is unsupported by frozen artifact schema")
        return self._raw_values([s.feature_values() for s in samples])
    def transform(self, samples: Sequence[ProcessSurrogateSample]) -> np.ndarray:
        if not hasattr(self, "scaler"): raise RuntimeError("preprocessor must be fit on training samples before transform")
        raw, masks = self._raw_samples(samples); return np.column_stack((self.scaler.transform(raw), masks))
    def transform_values(self, values: Sequence[Mapping[str, float]]) -> np.ndarray:
        if not hasattr(self, "scaler"): raise RuntimeError("preprocessor must be fit before transform")
        raw, masks = self._raw_values(values); return np.column_stack((self.scaler.transform(raw), masks))
    def state_fingerprint(self) -> str:
        hasher = _StateHasher("TrainOnlyPreprocessor:" + self.CONTRACT_VERSION); hasher.value("feature_names", self.feature_names); hasher.value("output_names", getattr(self, "output_names", ())); hasher.value("schema", self.schema.schema_fingerprint); _scaler_state(hasher, self.scaler, "scaler."); return hasher.digest()


def _baseline_fingerprint(model: object) -> str:
    h = _StateHasher(type(model).__module__ + "." + type(model).__qualname__)
    if isinstance(model, GaussianProcessBaseline):
        e = model.model; h.value("kind", "GaussianProcessBaseline"); h.value("kernel_class", type(e.kernel_).__qualname__); h.array("kernel_theta", e.kernel_.theta); h.array("X_train", e.X_train_); h.array("y_train", e.y_train_); h.array("alpha", e.alpha_); h.array("L", e.L_); h.array("y_train_mean", e._y_train_mean); h.array("y_train_std", e._y_train_std); h.value("normalize_y", e.normalize_y); h.value("n_features_in", e.n_features_in_); _scaler_state(h, model.scaler, "baseline_scaler."); return h.digest()
    if isinstance(model, TreeEnsembleBaseline):
        forest = model.model; h.value("kind", type(forest).__qualname__); h.value("n_estimators", len(forest.estimators_)); h.value("n_features_in", forest.n_features_in_)
        for index, estimator in enumerate(forest.estimators_):
            tree = estimator.tree_; h.value(f"tree.{index}.meta", {"node_count": tree.node_count, "max_depth": tree.max_depth})
            for name in ("children_left", "children_right", "feature", "threshold", "impurity", "n_node_samples", "weighted_n_node_samples", "value"): h.array(f"tree.{index}.{name}", getattr(tree, name))
        return h.digest()
    raise TypeError(f"unsupported surrogate baseline for state fingerprint: {type(model)!r}")


class ProcessSurrogate:
    def __init__(self, model_type: str = "gp", *, seed: int = 42) -> None:
        if model_type not in {"gp", "extra_trees"}: raise ValueError("model_type must be gp or extra_trees")
        self.model_type, self.seed = model_type, seed
    @property
    def uncertainty_kind(self) -> str: return "GP_POSTERIOR_STD" if self.model_type == "gp" else "TREE_ENSEMBLE_SPREAD"
    def fit(self, X: np.ndarray, targets: Mapping[str, np.ndarray]) -> "ProcessSurrogate":
        self.models = {}
        for target, values in targets.items():
            model = GaussianProcessBaseline(random_state=self.seed) if self.model_type == "gp" else TreeEnsembleBaseline("extra_trees", random_state=self.seed); self.models[target] = model.fit(X, np.asarray(values, dtype=float))
        if not self.models: raise ValueError("at least one target is required")
        self.model_state_fingerprint = self.state_fingerprint(); return self
    def state_fingerprint(self) -> str:
        if not hasattr(self, "models"): raise RuntimeError("unfitted surrogate cannot be fingerprinted")
        return _fingerprint({"contract": CONTRACT_VERSION, "model_type": self.model_type, "models": {name: _baseline_fingerprint(model) for name, model in sorted(self.models.items())}})
    def predict_distribution(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not hasattr(self, "models"): raise RuntimeError("surrogate must be fit before prediction")
        return {target: model.predict_distribution(X) for target, model in self.models.items()}


@dataclass
class SurrogateArtifact:
    surrogate: ProcessSurrogate; preprocessor: TrainOnlyPreprocessor; dataset_fingerprint: str; split_fingerprint: str; target_names: tuple[str, ...]; input_schema: SurrogateInputSchema; target_units: Mapping[str, str] = field(default_factory=dict); model_version: str = "process-surrogate-v2"; training_config: Mapping[str, Any] = field(default_factory=dict); metrics: Mapping[str, Any] = field(default_factory=dict); artifact_fingerprint: str = field(init=False); artifact_file_sha256: str | None = field(init=False, default=None)
    def __post_init__(self) -> None: self.artifact_fingerprint = self._computed_fingerprint()
    def _computed_fingerprint(self) -> str:
        return _fingerprint({"contract": CONTRACT_VERSION, "model_state": self.surrogate.state_fingerprint(), "preprocessor_state": self.preprocessor.state_fingerprint(), "input_schema": self.input_schema.schema_fingerprint, "dataset": self.dataset_fingerprint, "split": self.split_fingerprint, "targets": self.target_names, "target_units": dict(self.target_units), "training_config": _fingerprint(dict(self.training_config)), "model_version": self.model_version})
    def verify_integrity(self) -> None:
        if self.preprocessor.schema.schema_fingerprint != self.input_schema.schema_fingerprint: raise ValueError("artifact preprocessor schema differs from artifact schema")
        if self._computed_fingerprint() != self.artifact_fingerprint: raise ValueError("surrogate artifact fitted state integrity check failed")
    def predict(self, samples: Sequence[ProcessSurrogateSample]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.verify_integrity(); return self.surrogate.predict_distribution(self.preprocessor.transform(samples))
    def predict_decision(self, context: SurrogateDecisionContext, candidate_controls: Sequence[Mapping[str, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.verify_integrity()
        if context.stage.value != self.input_schema.supported_stage: raise ValueError("decision context stage is unsupported by artifact")
        expected = tuple(name.removeprefix("control::") for name in self.input_schema.control_features); rows = []
        for controls in candidate_controls:
            if tuple(sorted(controls)) != tuple(sorted(expected)): raise ValueError("candidate controls do not exactly match frozen legal control schema")
            rows.append({**context.feature_values(), **{f"control::{name}": float(value) for name, value in controls.items()}})
        return self.surrogate.predict_distribution(self.preprocessor.transform_values(rows))
    def metadata(self, *, artifact_file_sha256: str | None = None) -> dict[str, Any]:
        reference_present = "REFERENCE" in self.input_schema.fidelity_vocabulary
        reference_validation = self.metrics.get("reference_validation", {})
        reference_status = reference_validation.get("status", "NOT_EVALUATED" if reference_present else "NOT_AVAILABLE")
        return {"serializer": f"joblib:{joblib.__version__}", "model_type": self.surrogate.model_type, "uncertainty_kind": self.surrogate.uncertainty_kind, "model_version": self.model_version, "model_state_fingerprint": self.surrogate.state_fingerprint(), "preprocessor_state_fingerprint": self.preprocessor.state_fingerprint(), "dataset_fingerprint": self.dataset_fingerprint, "split_fingerprint": self.split_fingerprint, "target_schema": self.target_names, "target_units": dict(self.target_units), "input_schema": asdict(self.input_schema), "input_schema_fingerprint": self.input_schema.schema_fingerprint, "control_feature_names": self.input_schema.control_features, "context_feature_names": self.input_schema.context_features, "supported_stage": self.input_schema.supported_stage, "supported_fidelities": self.input_schema.fidelity_vocabulary, "reference_fidelity_present": reference_present, "reference_validation_status": reference_status, "training_config_fingerprint": _fingerprint(dict(self.training_config)), "metrics": self.metrics, "artifact_fingerprint": self.artifact_fingerprint, "artifact_file_sha256": artifact_file_sha256 or self.artifact_file_sha256}
    def save(self, path: str | Path) -> Path:
        self.verify_integrity(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path); digest = hashlib.sha256(path.read_bytes()).hexdigest(); self.artifact_file_sha256 = digest; path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(self.metadata(artifact_file_sha256=digest), indent=2, sort_keys=True), encoding="utf-8"); return path
    @classmethod
    def load(cls, path: str | Path, *, expected_dataset_fingerprint: str | None = None, expected_features: Sequence[str] | None = None, require_sidecar: bool = True) -> "SurrogateArtifact":
        path = Path(path); sidecar = path.with_suffix(path.suffix + ".metadata.json")
        if require_sidecar and not sidecar.is_file(): raise ValueError("artifact integrity sidecar is required")
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}; digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata and metadata.get("artifact_file_sha256") != digest: raise ValueError("artifact file SHA256 does not match integrity sidecar")
        artifact = joblib.load(path)
        if not isinstance(artifact, cls): raise TypeError("artifact is not a Process SurrogateArtifact")
        artifact.artifact_file_sha256 = digest; artifact.verify_integrity(); calculated = artifact.metadata(artifact_file_sha256=digest)
        for name in ("artifact_fingerprint", "model_state_fingerprint", "preprocessor_state_fingerprint", "input_schema_fingerprint", "dataset_fingerprint", "split_fingerprint"):
            if metadata and metadata.get(name) != calculated[name]: raise ValueError(f"artifact metadata disagrees with loaded fitted state: {name}")
        if expected_dataset_fingerprint and artifact.dataset_fingerprint != expected_dataset_fingerprint: raise ValueError("surrogate artifact dataset fingerprint mismatch")
        if expected_features and tuple(expected_features) != artifact.preprocessor.output_names: raise ValueError("surrogate artifact input schema mismatch")
        return artifact


def evaluate(artifact: SurrogateArtifact, samples: Sequence[ProcessSurrogateSample]) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted = artifact.predict(samples); rows: list[dict[str, object]] = []; metrics: dict[str, Any] = {}
    for target, (mean, std) in predicted.items():
        actual = np.asarray([sample.targets[target] for sample in samples], dtype=float); rows.extend({"sample_id": sample.sample_id, "target": target, "actual": float(y), "predicted": float(p), "uncertainty": float(s)} for sample, y, p, s in zip(samples, actual, mean, std)); safe = np.maximum(std, 1e-8); coverage = {str(level): float(np.mean(np.abs(actual - mean) <= z * safe)) for level, z in ((50, .67448975), (80, 1.28155157), (95, 1.95996398))}; item: dict[str, Any] = {"mae": float(mean_absolute_error(actual, mean)), "rmse": float(mean_squared_error(actual, mean) ** .5), "r2": float(r2_score(actual, mean)) if len(actual) > 1 and np.var(actual) else None, "uncertainty_kind": artifact.surrogate.uncertainty_kind}
        if artifact.surrogate.model_type == "gp": item.update({"gaussian_nll": float(np.mean(.5 * np.log(2 * np.pi * safe**2) + (actual - mean) ** 2 / (2 * safe**2))), "interval_coverage": coverage})
        else: item.update({"ensemble_spread": "UNSCALED_TREE_ENSEMBLE_SPREAD", "proxy_interval_coverage": coverage, "gaussian_proxy_nll": float(np.mean(.5 * np.log(2 * np.pi * safe**2) + (actual - mean) ** 2 / (2 * safe**2)))})
        metrics[target] = item
    return metrics, pd.DataFrame(rows)

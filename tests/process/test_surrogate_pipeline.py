from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.run_maspo_pipeline import synthetic_frame
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.stages import ProcessStage, STAGE_ORDER
from src.process.surrogates.core import ArtisticRunDirectoryAdapter, GenericTabularAdapter, ProcessSurrogate, SurrogateArtifact, SurrogateDecisionContext, SurrogateInputSchema, TrainOnlyPreprocessor, evaluate, split_groups
from src.process.surrogates.pipeline import ArtifactOptimizerBackend, PipelineConfig, _experiment_id, _reference_validation, run_pipeline


MAPPING = {
    "dataset_name": "synthetic-process-fixture", "dataset_version": "1", "sample_id": "sample_id", "run_id": "run_id", "recipe_id": "recipe_id", "group_id": "group_id", "stage": "stage", "fidelity": "fidelity", "requested_horizon": "requested_horizon",
    "controls": ["cbd_fraction", "calender_pressure"], "observations": ["drying_porosity"], "previous_state": ["slurry_viscosity"], "modality_state": [], "targets": ["capacity_retention"], "units": {"capacity_retention": "fraction"},
}


def _parts(model_type: str = "extra_trees"):
    adapter = GenericTabularAdapter(synthetic_frame(), MAPPING, source="synthetic-test")
    samples, manifest = adapter.samples(), adapter.manifest()
    split = split_groups(samples, manifest, seed=42)
    index = {sample.sample_id: sample for sample in samples}; train = [index[item] for item in split.train_ids]; test = [index[item] for item in split.test_ids]
    schema = SurrogateInputSchema.from_training_samples(train, declared_fidelities=manifest.fidelity_levels)
    preprocessor = TrainOnlyPreprocessor().fit(train, schema)
    surrogate = ProcessSurrogate(model_type, seed=42).fit(preprocessor.transform(train), {"capacity_retention": np.array([sample.targets["capacity_retention"] for sample in train])})
    return samples, manifest, split, train, test, SurrogateArtifact(surrogate, preprocessor, manifest.dataset_fingerprint, split.split_fingerprint, ("capacity_retention",), schema, {"capacity_retention": "fraction"})


def test_synthetic_pipeline_is_grouped_auditable_and_frozen(tmp_path) -> None:
    report = run_pipeline(GenericTabularAdapter(synthetic_frame(), MAPPING, source="synthetic-test"), tmp_path, PipelineConfig(("capacity_retention",), model_type="extra_trees", proposal_count=2))
    split = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8")); ids_to_group = {row.sample_id: row.group_id for row in GenericTabularAdapter(synthetic_frame(), MAPPING).samples()}
    groups = [{ids_to_group[item] for item in split[name]} for name in ("train_ids", "validation_ids", "test_ids")]
    assert report["status"] == "SUCCESS_SYNTHETIC_ONLY"
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all((tmp_path / name).is_file() for name in ("dataset_manifest.json", "validation_report.json", "split_manifest.json", "surrogate.joblib", "surrogate.joblib.metadata.json", "predictions.csv", "metrics.json", "proposals.json", "revalidation_queue.json", "experiment_manifest.json", "report.md"))
    assert "SOFTWARE TEST ONLY" in (tmp_path / "report.md").read_text(encoding="utf-8")
    proposals = json.loads((tmp_path / "proposals.json").read_text(encoding="utf-8"))
    for proposal in proposals:
        assert set(proposal["controls"]) == {"cbd_fraction", "calender_pressure"}
        assert "context_fingerprint" in proposal["provenance"]
    assert len({proposal["provenance"]["context_fingerprint"] for proposal in proposals}) == 1
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["offline_ranking"]["mode"] == "OFFLINE_RANKING_EVALUATION"
    assert set(metrics["latency"]["total_decision_ms"]) == {"p50", "p95", "p99"}
    assert report["reference_fidelity_present"] is False
    assert report["reference_validation_status"] == "NOT_AVAILABLE"


def test_reference_presence_does_not_claim_validation_without_heldout_evidence() -> None:
    _, manifest, split, train, _, absent = _parts()
    assert absent.metadata()["reference_fidelity_present"] is False
    assert absent.metadata()["reference_validation_status"] == "NOT_AVAILABLE"
    reference_train = [replace(sample, fidelity="REFERENCE") for sample in train]
    schema = SurrogateInputSchema.from_training_samples(reference_train, declared_fidelities=("REFERENCE",))
    preprocessor = TrainOnlyPreprocessor().fit(reference_train, schema)
    surrogate = ProcessSurrogate("extra_trees", seed=42).fit(preprocessor.transform(reference_train), {"capacity_retention": np.array([sample.targets["capacity_retention"] for sample in reference_train])})
    present = SurrogateArtifact(surrogate, preprocessor, manifest.dataset_fingerprint, split.split_fingerprint, ("capacity_retention",), schema, {"capacity_retention": "fraction"})
    assert present.metadata()["reference_fidelity_present"] is True
    assert present.metadata()["reference_validation_status"] == "NOT_EVALUATED"
    assert _reference_validation(reference_train, (), ())["status"] == "NOT_EVALUATED"
    assert _reference_validation(reference_train, (), (reference_train[0],))["status"] == "EVALUATED"


def test_experiment_id_binds_the_exact_surrogate_artifact() -> None:
    first = _experiment_id("dataset", "split", "config", "artifact-a")
    assert first == _experiment_id("dataset", "split", "config", "artifact-a")
    assert first != _experiment_id("dataset", "split", "config", "artifact-b")


def test_target_cannot_be_exposed_as_feature() -> None:
    report = GenericTabularAdapter(synthetic_frame(), {**MAPPING, "controls": ["capacity_retention"]}).validate()
    assert not report.valid and "target is incorrectly exposed as a feature: capacity_retention" in report.errors


@pytest.mark.parametrize("model_type", ["gp", "extra_trees"])
def test_fitted_model_state_binds_artifact_identity(model_type) -> None:
    _, manifest, split, train, _, first = _parts(model_type)
    schema = first.input_schema; pre = TrainOnlyPreprocessor().fit(train, schema); X = pre.transform(train); y = np.array([sample.targets["capacity_retention"] for sample in train])
    same = SurrogateArtifact(ProcessSurrogate(model_type, seed=42).fit(X, {"capacity_retention": y}), pre, manifest.dataset_fingerprint, split.split_fingerprint, ("capacity_retention",), schema, {"capacity_retention": "fraction"})
    different = SurrogateArtifact(ProcessSurrogate(model_type, seed=42).fit(X, {"capacity_retention": y + .01}), pre, manifest.dataset_fingerprint, split.split_fingerprint, ("capacity_retention",), schema, {"capacity_retention": "fraction"})
    assert first.surrogate.state_fingerprint() == same.surrogate.state_fingerprint()
    assert first.artifact_fingerprint == same.artifact_fingerprint != different.artifact_fingerprint


def test_artifact_detects_fitted_state_mutation_and_file_tampering(tmp_path) -> None:
    _, _, _, _, test, artifact = _parts()
    artifact.preprocessor.scaler.mean_[0] += .1
    with pytest.raises(ValueError, match="integrity"): artifact.verify_integrity()
    _, _, _, _, test, artifact = _parts(); artifact.surrogate.models["capacity_retention"] = ProcessSurrogate("extra_trees").fit(artifact.preprocessor.transform(test), {"capacity_retention": np.array([sample.targets["capacity_retention"] for sample in test])}).models["capacity_retention"]
    with pytest.raises(ValueError, match="integrity"): artifact.verify_integrity()
    _, _, _, _, test, artifact = _parts(); path = artifact.save(tmp_path / "surrogate.joblib"); path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"): SurrogateArtifact.load(path)


def test_save_load_predictions_and_schema_guards(tmp_path) -> None:
    _, _, _, _, test, artifact = _parts(); before = artifact.predict(test)["capacity_retention"][0]; path = artifact.save(tmp_path / "surrogate.joblib")
    loaded = SurrogateArtifact.load(path, expected_dataset_fingerprint=artifact.dataset_fingerprint, expected_features=artifact.preprocessor.output_names)
    assert np.allclose(before, loaded.predict(test)["capacity_retention"][0])
    with pytest.raises(ValueError, match="dataset"): SurrogateArtifact.load(path, expected_dataset_fingerprint="wrong")
    with pytest.raises(ValueError, match="input schema"): SurrogateArtifact.load(path, expected_features=("wrong",))


def test_train_only_scaler_and_unknown_structural_features_fail_closed() -> None:
    _, _, _, train, test, artifact = _parts(); raw = np.array([[sample.feature_values().get(name, 0.) for name in artifact.preprocessor.feature_names] for sample in train])
    assert np.allclose(artifact.preprocessor.scaler.mean_, raw.mean(axis=0))
    unseen_fidelity = replace(test[0], fidelity="REFERENCE")
    with pytest.raises(ValueError, match="unknown structural"): artifact.predict([unseen_fidelity])
    with pytest.raises(ValueError, match="unsupported"): artifact.predict([replace(test[0], stage=ProcessStage.DRYING)])
    missing_known = replace(test[0], controls={"cbd_fraction": test[0].controls["cbd_fraction"]})
    transformed = artifact.preprocessor.transform([missing_known]); index = artifact.preprocessor.output_names.index("control::calender_pressure__observed")
    assert transformed[0, index] == 0.


def test_fixed_context_maspo_exposes_only_controls() -> None:
    _, _, _, train, test, artifact = _parts(); sample = test[0]
    context = SurrogateDecisionContext(sample.stage, sample.previous_state, sample.observations, sample.modality_state, sample.fidelity, sample.requested_horizon, {"run": sample.run_id})
    pool = pd.DataFrame([{"recipe_id": "a", "cbd_fraction": .03, "calender_pressure": 40.}, {"recipe_id": "b", "cbd_fraction": .05, "calender_pressure": 55.}])
    backend = ArtifactOptimizerBackend(artifact, context); proposal = backend.propose([], pool, "capacity_retention")[0]
    assert set(proposal.design_variables) == {"cbd_fraction", "calender_pressure"}
    assert proposal.metadata["context_fingerprint"] == context.context_fingerprint
    with pytest.raises(ValueError, match="candidate pool includes"): backend.propose([], pool.assign(**{"previous::slurry_viscosity": 1.}), "capacity_retention")


def test_tree_and_gp_uncertainty_labels_and_metrics() -> None:
    for model_type, expected in (("gp", "GP_POSTERIOR_STD"), ("extra_trees", "TREE_ENSEMBLE_SPREAD")):
        _, _, _, _, test, artifact = _parts(model_type); metrics, _ = evaluate(artifact, test); item = metrics["capacity_retention"]
        assert item["uncertainty_kind"] == expected
        assert ("gaussian_nll" in item) == (model_type == "gp")
        assert ("gaussian_proxy_nll" in item) == (model_type == "extra_trees")


def test_generic_validation_rejects_missing_columns_nonfinite_and_invalid_units() -> None:
    assert not GenericTabularAdapter(synthetic_frame(), {**MAPPING, "stage": "missing"}).validate().valid
    bad = synthetic_frame(); bad.loc[0, "cbd_fraction"] = np.inf
    assert not GenericTabularAdapter(bad, MAPPING).validate().valid
    assert not GenericTabularAdapter(synthetic_frame(), {**MAPPING, "unit_schema": {"capacity_retention": ["percent"]}}).validate().valid


def test_artistic_adapter_preserves_physics_provenance_and_rejects_nonphysical_label(tmp_path, monkeypatch) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    provenance = ProvenanceRecord("SIMULATED_STRESS", processing_parameters={"recipe_fingerprint": "recipe-hash", "simulation_manifest_sha256": "manifest-hash", "physics_config_fingerprint": "physics-hash", "fidelity_mode": "SHORT_HORIZON", "requested_slurry_steps": 500_000})
    stage = StageRecord("cal", ProcessStage.CALENDERING, STAGE_ORDER[ProcessStage.CALENDERING], {"pressure": ParameterValue(50.)}, {"porosity": MeasurementValue(.4)}, [], provenance=provenance)
    run = BatteryProcessRun("run-1", None, "recipe-hash", "ARTISTIC_NMC", {}, {}, [stage], {"capacity_retention": MeasurementValue(.8)}, provenance)
    adapter = ArtisticRunDirectoryAdapter(tmp_path, targets=("capacity_retention",)); monkeypatch.setattr(adapter, "_runs", lambda: [run])
    sample = adapter.samples()[0]
    assert sample.group_id == "recipe-hash" and sample.physics_config_fingerprint == "physics-hash" and sample.fidelity == "SHORT_HORIZON" and sample.provenance["simulation_manifest_sha256"] == "manifest-hash"
    rejected = replace(run, provenance=replace(provenance, evidence_kind="FAILED")); monkeypatch.setattr(adapter, "_runs", lambda: [rejected])
    assert not adapter.validate().valid

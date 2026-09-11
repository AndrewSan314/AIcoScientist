from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.simulators.artistic.config import PINNED_COMMIT, PINNED_SOURCE_TREE_HASH, SOURCE_URL
from src.process.simulators.artistic.schemas import ArtisticRecipe, DryingMode, recipe_fingerprint
from src.process.simulators.base import SimulationResult, SimulationStatus
from src.process.stages import ProcessStage, STAGE_ORDER


class ArtisticSimulationAdapter(NormalizedRunAdapter):
    ADAPTER_VERSION = "2"
    SCHEMA_VERSION = "2"

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="artistic", display_name="ARTISTIC Physics Stress", chemistry="lithium-ion electrode simulation",
            evidence_kind="SIMULATED_PHYSICS", source_doi="not applicable: pinned public GitHub source", version=PINNED_COMMIT, license="CC BY-NC-SA 4.0",
            process_stages=(ProcessStage.MIXING, ProcessStage.DRYING, ProcessStage.CALENDERING), modalities=("PROCESS_TABULAR",),
            recommended_splits=("OOD_FACTOR_EXTREME",), optimization_capable=True, multimodal_capable=False,
            limitations="Runs are public-source simulated physics, never physical observations; training is blocked until a validated real execution exists.")

    @staticmethod
    def from_simulation(result: SimulationResult, recipe: ArtisticRecipe) -> BatteryProcessRun:
        if result.status != SimulationStatus.SUCCESS:
            raise ValueError(f"ARTISTIC result is not valid simulated physics: {result.status}")
        if result.provenance.get("checked_out_commit") != PINNED_COMMIT or result.provenance.get("source_tree_hash") != PINNED_SOURCE_TREE_HASH:
            raise ValueError("ARTISTIC simulation provenance is not pinned to the audited source")
        provenance = ProvenanceRecord(
            evidence_kind="SIMULATED_PHYSICS", source_url=SOURCE_URL, source_version=PINNED_COMMIT,
            raw_hashes=dict(result.provenance.get("rendered_source_file_hashes", {})), adapter_version="2",
            processing_parameters={
                "simulation_manifest": str(result.run_directory / "manifest.json"), "simulation_manifest_sha256": result.provenance.get("simulation_manifest_sha256"),
                "status": str(result.status), "source_commit": result.provenance.get("checked_out_commit"), "source_tree_hash": result.provenance.get("source_tree_hash"),
                "patches": result.provenance.get("patches", []), "rendered_input_hashes": dict(result.provenance.get("rendered_source_file_hashes", {})),
                "physics_output_hashes": dict(result.provenance.get("output_hashes", {})), "stage_lineage": result.provenance.get("stage_lineage", []),
                "recipe_fingerprint": recipe_fingerprint(recipe), "executable_versions": dict(result.provenance.get("executable_versions", {})),
                "executable_identity": dict(result.provenance.get("executable_identity", {})),
            },
        )
        stages: list[StageRecord] = [
            _stage("artistic-mixing", ProcessStage.MIXING, _slurry_controls(recipe), result.stage_outputs.get("slurry", {}), None, provenance),
        ]
        if recipe.drying_mode:
            stages.append(_stage("artistic-drying", ProcessStage.DRYING, _drying_controls(recipe), result.stage_outputs.get("drying", {}), "artistic-mixing", provenance))
        if recipe.calendering:
            stages.append(_stage("artistic-calendering", ProcessStage.CALENDERING, _calendering_controls(recipe), result.stage_outputs.get("calendering", {}), "artistic-drying", provenance))
        run = BatteryProcessRun(
            run_id=result.run_id, cell_id=None, batch_id=recipe_fingerprint(recipe), chemistry_id="ARTISTIC_NMC",
            equipment_context={"simulator": "LAMMPS", "runner": result.provenance.get("commands", [])}, environment_context={"executable_versions": result.provenance.get("executable_versions", {}), "executable_identity": result.provenance.get("executable_identity", {})}, stages=stages,
            final_kpis={name: MeasurementValue(value, source_name=name) for name, value in result.final_outputs.items()}, provenance=provenance,
        )
        visible = {name for stage in run.stages for name in stage.intermediate_properties}
        overlap = visible & set(run.final_kpis)
        if overlap:
            raise ValueError(f"ARTISTIC final target leakage: {sorted(overlap)}")
        return run

    @classmethod
    def normalize_successful(cls, result: SimulationResult, recipe: ArtisticRecipe, *, root: str | Path | None = None) -> Path:
        """Persist only a successful, source-pinned simulation for adapter and BPSS use."""
        if result.status != SimulationStatus.SUCCESS:
            cls.from_simulation(result, recipe)
        adapter = cls(root)
        source_manifest_path = result.run_directory / "manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("successful ARTISTIC run lacks its simulation manifest")
        source_manifest_bytes = source_manifest_path.read_bytes()
        try:
            source_manifest = json.loads(source_manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError("successful ARTISTIC simulation manifest is invalid JSON") from exc
        if not isinstance(source_manifest, dict) or source_manifest.get("checked_out_commit") != PINNED_COMMIT or source_manifest.get("source_tree_hash") != PINNED_SOURCE_TREE_HASH:
            raise ValueError("successful ARTISTIC simulation manifest is not pinned to the audited source")
        manifest_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
        manifest_rel = Path("raw") / "simulation_manifests" / f"{result.run_id}-{manifest_hash}.json"
        provenance = {**result.provenance, "simulation_manifest_sha256": manifest_hash, "normalized_simulation_manifest": manifest_rel.as_posix()}
        result = replace(result, provenance=provenance)
        run = cls.from_simulation(result, recipe)
        adapter.root.mkdir(parents=True, exist_ok=True)
        existing = adapter.load_runs() if adapter.normalized_runs_path.is_file() else []
        if any(item.run_id == run.run_id for item in existing):
            raise ValueError(f"ARTISTIC run is already normalized: {run.run_id}")
        hashes = dict(result.provenance.get("rendered_source_file_hashes", {}))
        if not hashes:
            raise ValueError("successful ARTISTIC run lacks source hashes")
        manifest_path = adapter.root / manifest_rel
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and manifest_path.read_bytes() != source_manifest_bytes:
            raise ValueError(f"immutable ARTISTIC manifest collision: {manifest_path.name}")
        if not manifest_path.exists():
            shutil.copyfile(source_manifest_path, manifest_path)
        aggregate_path = adapter.root / "manifest.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else {}
        entries = aggregate.get("normalized_runs", [])
        if not isinstance(entries, list):
            raise ValueError("ARTISTIC aggregate manifest has invalid normalized_runs")
        entry = {
            "run_id": run.run_id, "recipe_fingerprint": recipe_fingerprint(recipe),
            "simulation_manifest": manifest_rel.as_posix(), "simulation_manifest_sha256": manifest_hash,
            "source_simulation_manifest": str(source_manifest_path), "source_commit": result.provenance.get("checked_out_commit"),
            "source_tree_hash": result.provenance.get("source_tree_hash"), "patches": result.provenance.get("patches", []),
            "rendered_input_hashes": hashes, "physics_output_hashes": result.provenance.get("output_hashes", {}),
            "stage_lineage": result.provenance.get("stage_lineage", []), "executable_versions": result.provenance.get("executable_versions", {}),
            "executable_identity": result.provenance.get("executable_identity", {}),
        }
        aggregate = {"official_dataset_source": SOURCE_URL, "license": "CC BY-NC-SA 4.0", "pinned_upstream_commit": PINNED_COMMIT, "source_tree_hash": PINNED_SOURCE_TREE_HASH, "normalized_runs": [*entries, entry]}
        aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
        all_hashes = _aggregate_hashes(aggregate["normalized_runs"])
        return adapter.write_processed_cache([*existing, run], raw_hashes=all_hashes)


def _aggregate_hashes(entries: list[object]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("ARTISTIC aggregate manifest entry is invalid")
        run_id = str(item["run_id"])
        hashes[f"{run_id}/simulation_manifest.json"] = str(item["simulation_manifest_sha256"])
        for category in ("rendered_input_hashes", "physics_output_hashes"):
            for name, digest in dict(item.get(category, {})).items():
                hashes[f"{run_id}/{category}/{name}"] = str(digest)
    return hashes


def _stage(stage_id: str, stage_type: ProcessStage, controls: dict[str, object], properties: dict[str, float], upstream: str | None, provenance: ProvenanceRecord) -> StageRecord:
    return StageRecord(
        stage_id=stage_id, stage_type=stage_type, sequence_index=STAGE_ORDER[stage_type], upstream_stage_id=upstream,
        controls={name: ParameterValue(value, source_name=name) for name, value in controls.items()},
        intermediate_properties={name: MeasurementValue(value, source_name=name) for name, value in properties.items()}, modalities=[], provenance=provenance,
    )


def _slurry_controls(recipe: ArtisticRecipe) -> dict[str, object]:
    return dict(recipe.slurry.template_values())


def _drying_controls(recipe: ArtisticRecipe) -> dict[str, object]:
    return dict(recipe.heterogeneous_drying.template_values()) if recipe.drying_mode == DryingMode.HETEROGENEOUS and recipe.heterogeneous_drying else {"drying_mode": DryingMode.HOMOGENEOUS.value}


def _calendering_controls(recipe: ArtisticRecipe) -> dict[str, object]:
    return {"compression_degree": recipe.calendering.compression_degree, "cbd_nanoporosity_decrease": recipe.calendering.cbd_nanoporosity_decrease, "relaxation": recipe.calendering.relaxation, "perform_energy_minimization": recipe.calendering.perform_energy_minimization} if recipe.calendering else {}

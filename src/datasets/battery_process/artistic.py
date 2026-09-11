from __future__ import annotations

import json
from pathlib import Path

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.simulators.artistic.config import PINNED_COMMIT, PINNED_SOURCE_TREE_HASH, SOURCE_URL
from src.process.simulators.artistic.schemas import ArtisticRecipe, DryingMode
from src.process.simulators.base import SimulationResult, SimulationStatus
from src.process.stages import ProcessStage, STAGE_ORDER


class ArtisticSimulationAdapter(NormalizedRunAdapter):
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
            processing_parameters={"manifest": str(result.run_directory / "manifest.json"), "status": str(result.status), "patches": result.provenance.get("patches", [])},
        )
        stages: list[StageRecord] = [
            _stage("artistic-mixing", ProcessStage.MIXING, _slurry_controls(recipe), result.stage_outputs.get("slurry", {}), None, provenance),
        ]
        if recipe.drying_mode:
            stages.append(_stage("artistic-drying", ProcessStage.DRYING, _drying_controls(recipe), result.stage_outputs.get("drying", {}), "artistic-mixing", provenance))
        if recipe.calendering:
            stages.append(_stage("artistic-calendering", ProcessStage.CALENDERING, _calendering_controls(recipe), result.stage_outputs.get("calendering", {}), "artistic-drying", provenance))
        run = BatteryProcessRun(
            run_id=result.run_id, cell_id=None, batch_id=result.run_id, chemistry_id="ARTISTIC_NMC",
            equipment_context={"simulator": "LAMMPS", "runner": result.provenance.get("commands", [])}, environment_context={}, stages=stages,
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
        adapter = cls(root)
        run = cls.from_simulation(result, recipe)
        adapter.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "official_dataset_source": SOURCE_URL, "license": "CC BY-NC-SA 4.0",
            "pinned_upstream_commit": PINNED_COMMIT, "source_tree_hash": PINNED_SOURCE_TREE_HASH,
            "simulation_manifest": str(result.run_directory / "manifest.json"),
        }
        (adapter.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        existing = adapter.load_runs() if adapter.normalized_runs_path.is_file() else []
        if any(item.run_id == run.run_id for item in existing):
            raise ValueError(f"ARTISTIC run is already normalized: {run.run_id}")
        hashes = dict(result.provenance.get("rendered_source_file_hashes", {}))
        if not hashes:
            raise ValueError("successful ARTISTIC run lacks source hashes")
        return adapter.write_processed_cache([*existing, run], raw_hashes=hashes)


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

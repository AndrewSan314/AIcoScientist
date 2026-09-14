from src.datasets.battery_process.base import BatteryDatasetMetadata, ProcessPredictionTask
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage
from src.process.surrogates.battery_adapter import BatteryProcessSurrogateAdapter, registered_battery_datasets


class _Source:
    def metadata(self):
        return BatteryDatasetMetadata("fixture", "fixture", "NMC", "PHYSICAL_HISTORICAL", "doi", "1", "CC", (ProcessStage.FORMULATION, ProcessStage.MIXING, ProcessStage.COATING), ("ULTRASOUND_SIGNAL",), ("GROUP_BY_PROCESS_RECIPE",), True, True, "fixture")

    def load_runs(self):
        provenance = ProvenanceRecord("PHYSICAL_HISTORICAL", source_doi="doi", raw_hashes={"raw.csv": "abc"})
        return [BatteryProcessRun(f"run-{index}", None, f"batch-{index}", "NMC", {}, {}, [StageRecord(f"f-{index}", ProcessStage.FORMULATION, 0, {"solids": ParameterValue(index + 1)}, {}, [], provenance=provenance), StageRecord(f"m-{index}", ProcessStage.MIXING, 1, {"speed": ParameterValue(index + 2)}, {}, [ModalityObservation(f"u-{index}", ModalityType.ULTRASOUND_SIGNAL, ProcessStage.MIXING, missing_reason="not_collected")], f"f-{index}", provenance=provenance), StageRecord(f"c-{index}", ProcessStage.COATING, 2, {"gap": ParameterValue(index + 3)}, {}, [], f"m-{index}", provenance=provenance)], {"cell_capacity_mah": MeasurementValue(10 + index, "mAh")}, provenance) for index in range(3)]

    def validate(self):
        from src.process.validation import ProcessValidationReport
        return ProcessValidationReport(valid=True)


def test_bridge_preserves_source_identity_and_horizon_boundary():
    bridge = BatteryProcessSurrogateAdapter(_Source(), task=ProcessPredictionTask("cell_capacity_mah", ProcessStage.COATING))
    sample = bridge.samples()[0]
    assert "coating.gap" not in sample.controls
    assert sample.modality_state["mixing.ultrasound_signal.observed"] == 0.0
    assert sample.provenance["raw_hashes"] == {"raw.csv": "abc"}
    assert bridge.manifest().source_hashes["run-0:raw.csv"] == "abc"
    assert bridge.validate().valid


def test_registered_dataset_vocabulary_is_fixed():
    assert registered_battery_datasets() == ("artistic", "drakopoulos_graphite", "naion_hte", "warwick_nmc622", "warwick_ultrasound")

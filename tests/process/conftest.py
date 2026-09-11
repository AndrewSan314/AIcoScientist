from __future__ import annotations

from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage


def process_run() -> BatteryProcessRun:
    provenance = ProvenanceRecord(evidence_kind="PHYSICAL_HISTORICAL", raw_hashes={"raw.csv": "abc"})
    formulation = StageRecord("form", ProcessStage.FORMULATION, 0, {"solids": ParameterValue(0.6)}, {}, [], provenance=provenance)
    mixing = StageRecord("mix", ProcessStage.MIXING, 1, {"speed": ParameterValue(100)}, {"viscosity": MeasurementValue(5)}, [ModalityObservation("mix-tab", ModalityType.PROCESS_TABULAR, ProcessStage.MIXING, values=[100, 5], provenance=provenance)], "form", provenance=provenance)
    coating = StageRecord("coat", ProcessStage.COATING, 2, {"gap": ParameterValue(50)}, {"wet_thickness": MeasurementValue(70)}, [], "mix", provenance=provenance)
    drying = StageRecord("dry", ProcessStage.DRYING, 3, {"temperature": ParameterValue(100)}, {"dry_thickness": MeasurementValue(50)}, [], "coat", provenance=provenance)
    calender = StageRecord("cal", ProcessStage.CALENDERING, 4, {"pressure": ParameterValue(20)}, {"porosity": MeasurementValue(0.35)}, [], "dry", provenance=provenance)
    return BatteryProcessRun("run-1", "cell-1", "batch-a", "NMC622", {}, {}, [formulation, mixing, coating, drying, calender], {"capacity": MeasurementValue(150), "impedance": MeasurementValue(4)}, provenance)

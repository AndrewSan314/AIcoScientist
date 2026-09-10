from __future__ import annotations

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.stages import ProcessStage


class WarwickUltrasoundAdapter(NormalizedRunAdapter):
    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="warwick_ultrasound", display_name="Warwick Ultrasonic Inline QC", chemistry="graphite anode / NMC622 cathode",
            evidence_kind="PHYSICAL_HISTORICAL", source_doi="10.17632/c62yn37d9h.1", version="1", license="CC BY 4.0",
            process_stages=(ProcessStage.CALENDERING,), modalities=("ULTRASOUND_SIGNAL", "ULTRASOUND_SPECTRUM", "PROCESS_TABULAR"),
            recommended_splits=("LEAVE_ONE_PROCESS_SETTING_OUT",), optimization_capable=False, multimodal_capable=True,
            limitations="Post-calender signals must not be used before calendering; raw schema requires audit.")

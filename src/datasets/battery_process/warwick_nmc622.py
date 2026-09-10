from __future__ import annotations

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.stages import ProcessStage


class WarwickNMC622Adapter(NormalizedRunAdapter):
    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="warwick_nmc622", display_name="Warwick NMC622 Pilot Multimodal", chemistry="NMC622 Li-ion electrode/cell",
            evidence_kind="PILOT_LINE_HISTORICAL", source_doi="10.17632/wwhm2frfmy.1", version="1", license="CC0 1.0",
            process_stages=(ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("PROCESS_TABULAR", "SCALAR_METROLOGY", "EIS_SPECTRUM", "CYCLING_CURVE"), recommended_splits=("GROUP_BY_DOE_CONDITION",),
            optimization_capable=True, multimodal_capable=True, limitations="Replicate grouping and source assets require audit before use.")

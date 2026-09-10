from __future__ import annotations

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.stages import ProcessStage


class DrakopoulosGraphiteAdapter(NormalizedRunAdapter):
    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="drakopoulos_graphite", display_name="Graphite Process-15", chemistry="graphite Li-ion electrode",
            evidence_kind="PHYSICAL_HISTORICAL", source_doi="10.17632/4dh2h3tsf4.1", version="1", license="CC BY 4.0",
            process_stages=(ProcessStage.FORMULATION, ProcessStage.MIXING, ProcessStage.COATING, ProcessStage.DRYING, ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("PROCESS_TABULAR", "SCALAR_METROLOGY"), recommended_splits=("GROUP_BY_PROCESS_RECIPE", "OOD_FACTOR_EXTREME"),
            optimization_capable=True, multimodal_capable=False, limitations="Raw files and source schema require audit before use.")

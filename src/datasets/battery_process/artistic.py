from __future__ import annotations

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.stages import ProcessStage


class ArtisticSimulationAdapter(NormalizedRunAdapter):
    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="artistic", display_name="ARTISTIC Physics Stress", chemistry="lithium-ion electrode simulation",
            evidence_kind="SIMULATED_PHYSICS", source_doi="10.5281/zenodo.5956128", version="5956128", license="CC BY-NC-SA 4.0",
            process_stages=(ProcessStage.MIXING, ProcessStage.DRYING, ProcessStage.CALENDERING), modalities=("PROCESS_TABULAR", "XCT_VOLUME"),
            recommended_splits=("OOD_FACTOR_EXTREME",), optimization_capable=True, multimodal_capable=True,
            limitations="Zenodo record is source-audited but its simulation files are restricted; simulated evidence only.")

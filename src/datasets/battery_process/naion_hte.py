from __future__ import annotations

from .base import BatteryDatasetMetadata, NormalizedRunAdapter
from src.process.stages import ProcessStage


class NaIonHTEAdapter(NormalizedRunAdapter):
    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="naion_hte", display_name="Na-ion High-Throughput Process Chain", chemistry="sodium-ion",
            evidence_kind="PHYSICAL_HISTORICAL", source_doi="10.5281/zenodo.7981011", version="unverified", license="unverified",
            process_stages=(ProcessStage.FORMULATION, ProcessStage.COATING, ProcessStage.ASSEMBLY, ProcessStage.FORMATION, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("PROCESS_TABULAR", "FORMATION_CURVE", "CYCLING_CURVE"), recommended_splits=("GROUP_BY_BATCH", "LATER_RUN_HOLDOUT"),
            optimization_capable=False, multimodal_capable=True, limitations="Archive, license and source mapping require audit before use.")

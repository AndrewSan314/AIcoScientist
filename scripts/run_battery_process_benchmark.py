from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter


ADAPTERS = [DrakopoulosGraphiteAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter, NaIonHTEAdapter, ArtisticSimulationAdapter]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-backed BPSS benchmark manifest.")
    parser.add_argument("--output", type=Path, default=Path("outputs/process_benchmark"))
    parser.add_argument("--allow-unavailable", action="store_true", help="Write an audit-only manifest when raw source data have not been acquired.")
    args = parser.parse_args()
    root = args.output
    for directory in ("dataset_audits", "prediction", "calibration", "multimodal_ablations", "missing_modality", "stage_ablations", "optimization", "stress", "figures"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    audits = []
    unavailable = []
    for adapter_class in ADAPTERS:
        adapter = adapter_class()
        report = adapter.validate()
        audit = {"metadata": adapter.metadata().__dict__, "validation": {"valid": report.valid, "errors": list(report.errors)}}
        dataset_id = adapter.metadata().dataset_id
        (root / "dataset_audits" / f"{dataset_id}.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        audits.append(audit)
        if not report.valid:
            unavailable.append(dataset_id)
    manifest = {"suite": "BPSS", "datasets": audits, "status": "AUDIT_ONLY" if unavailable else "READY", "unavailable": unavailable}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (root / "PROCESS_BENCHMARK_REPORT.md").write_text(
        "# Battery Process Stress Suite\n\n" + ("Raw source data are not yet auditable: " + ", ".join(unavailable) if unavailable else "All registered adapters passed source validation."),
        encoding="utf-8",
    )
    if unavailable and not args.allow_unavailable:
        raise SystemExit("BPSS refused to benchmark unaudited source data; rerun with --allow-unavailable for an audit-only manifest.")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

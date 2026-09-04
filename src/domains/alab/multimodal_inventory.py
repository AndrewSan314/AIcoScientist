from __future__ import annotations

import json
import os
import zipfile
from typing import Any

from src.domains.alab.artifact_index import ALabArtifactIndex


def _archive_inventory(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"present": False, "raw_artifact_count": 0, "members": []}
    with zipfile.ZipFile(path) as archive:
        members = [info.filename for info in archive.infolist() if not info.is_dir()]
    return {
        "present": True,
        "raw_artifact_count": len(members),
        "members": members,
    }


def inventory_alab_modalities(
    data_dir: str = "data/external/precursor_genome_2026",
    cache_dir: str = "data/derived/alab",
) -> dict[str, Any]:
    ledger_path = os.path.join(data_dir, "ledger_precursor_genome.json")
    samples: list[dict[str, Any]] = []
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        samples = payload.get("samples", []) if isinstance(payload, dict) else payload
    sample_ids = {str(sample.get("sample_id")) for sample in samples if sample.get("sample_id")}
    index = ALabArtifactIndex(data_dir=data_dir, cache_dir=cache_dir).build_or_load(samples=samples)
    archive_counts = {
        name: _archive_inventory(os.path.join(data_dir, name))
        for name in ("raw_scans.zip", "refinement_pkls.zip", "sem.zip", "eds.zip")
    }

    linked = {
        modality: {sid for sid in sample_ids if index.has_artifact(sid, modality)}
        for modality in ("XRD", "REFINEMENT")
    }
    outcome_ids = {
        str(sample.get("sample_id"))
        for sample in samples
        if sample.get("sample_id") and (sample.get("outcome") or {}).get("reaction_category") not in (None, "none")
    }
    sem = _archive_inventory(os.path.join(data_dir, "sem.zip"))
    eds = _archive_inventory(os.path.join(data_dir, "eds.zip"))

    def precursor_coverage(archive: dict[str, Any], modality: str) -> dict[str, Any]:
        precursor_ids = sorted({part for member in archive["members"] for part in member.split("/") if part.startswith("precursor_")})
        return {
            "modality": modality,
            "archive_present": archive["present"],
            "raw_artifact_count": archive["raw_artifact_count"],
            "archive_precursor_group_count": len(precursor_ids),
            "linked_candidate_samples": 0,
            "candidate_sample_linkage_quality": "precursor_level_only_unlinked_to_sample_id",
            "derived_observable_coverage": 0,
            "missingness": "candidate linkage unavailable; excluded from candidate×modality replay",
            "action_space_supported": False,
        }

    modalities = {
        "XRD": {
            "archive_present": os.path.exists(os.path.join(data_dir, "raw_scans.zip")),
            "raw_artifact_count": archive_counts["raw_scans.zip"]["raw_artifact_count"],
            "linked_candidate_samples": len(linked["XRD"]),
            "candidate_sample_linkage_quality": "canonical_sample_linked",
            "derived_observable_coverage": len(linked["XRD"]),
            "missingness": f"{len(sample_ids - linked['XRD'])} ledger samples without canonical XRD artifact",
            "action_space_supported": True,
        },
        "REFINEMENT": {
            "archive_present": os.path.exists(os.path.join(data_dir, "refinement_pkls.zip")),
            "raw_artifact_count": archive_counts["refinement_pkls.zip"]["raw_artifact_count"],
            "linked_candidate_samples": len(linked["REFINEMENT"]),
            "candidate_sample_linkage_quality": "canonical_sample_linked",
            "derived_observable_coverage": len(linked["REFINEMENT"]),
            "missingness": f"{len(sample_ids - linked['REFINEMENT'])} ledger samples without canonical refinement artifact",
            "action_space_supported": True,
        },
        "SEM": precursor_coverage(sem, "SEM"),
        "EDS": precursor_coverage(eds, "EDS"),
        "OUTCOME_TEST": {
            "archive_present": True,
            "raw_artifact_count": 0,
            "linked_candidate_samples": len(outcome_ids),
            "candidate_sample_linkage_quality": "ledger_sample_linked",
            "derived_observable_coverage": len(outcome_ids),
            "missingness": f"{len(sample_ids - outcome_ids)} samples without classified outcome",
            "action_space_supported": True,
        },
    }
    return {
        "dataset": "AmanchukwuLab/AL-anode-free precursor_genome_2026",
        "data_dir": data_dir,
        "available_samples": len(sample_ids),
        "modalities": modalities,
        "archives": {
            name: {"path": os.path.join(data_dir, name), **archive_counts[name]}
            for name in archive_counts
        },
        "linkage_rule": "Only canonical sample_id links are eligible for candidate×modality actions; precursor-level SEM/EDS members are inventory-only.",
    }


__all__ = ["inventory_alab_modalities"]

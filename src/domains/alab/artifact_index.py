from __future__ import annotations

import json
import logging
import os
import zipfile
from dataclasses import asdict, dataclass
from typing import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRef:
    """Lightweight reference pointing to an artifact stored inside an external archive."""

    archive_path: str
    member_path: str
    modality: str
    size_bytes: int | None = None
    checksum: str | None = None
    selected_scan_index: int | None = None
    selected_case_index: int | None = None
    selection_method: str = "canonical"
    is_canonical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRef:
        return cls(
            archive_path=str(data["archive_path"]),
            member_path=str(data["member_path"]),
            modality=str(data["modality"]),
            size_bytes=int(data["size_bytes"]) if data.get("size_bytes") is not None else None,
            checksum=str(data["checksum"]) if data.get("checksum") is not None else None,
            selected_scan_index=int(data["selected_scan_index"]) if data.get("selected_scan_index") is not None else None,
            selected_case_index=int(data["selected_case_index"]) if data.get("selected_case_index") is not None else None,
            selection_method=str(data.get("selection_method", "canonical")),
            is_canonical=bool(data.get("is_canonical", True)),
        )


@dataclass(frozen=True)
class ResolvedArtifactSelection:
    """Explicit resolved artifact selection contract linking metadata to artifact bytes."""

    sample_id: str
    requested_modality: str
    selected_scan_index: int | None
    selected_case_index: int | None
    artifact_ref: ArtifactRef | None
    selection_method: str
    is_canonical: bool


class ALabArtifactIndex:
    """Zero-extraction streaming index for external A-Lab ZIP archives."""

    def __init__(
        self,
        data_dir: str = "data/external/precursor_genome_2026",
        cache_dir: str = "data/derived/alab",
    ) -> None:
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self._index: dict[str, dict[str, ArtifactRef]] = {}
        self._initialized = False

    def _get_manifest_checksums(self) -> dict[str, str]:
        """Loads trusted archive checksums from manifest if available."""
        manifest_paths = [
            os.path.join(os.path.dirname(self.data_dir), "aicoscientist_datasets_manifest.json"),
            "data/external/aicoscientist_datasets_manifest.json",
        ]
        for mp in manifest_paths:
            if os.path.exists(mp):
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    files_dict = manifest.get("datasets", {}).get("precursor_genome_2026", {}).get("files", {})
                    return {
                        "raw_scans.zip": files_dict.get("raw_scans.zip", {}).get("md5", ""),
                        "refinement_pkls.zip": files_dict.get("refinement_pkls.zip", {}).get("md5", ""),
                    }
                except Exception:
                    pass
        return {}

    def _get_archive_stat_signature(self) -> dict[str, Any]:
        """Returns archive size/mtime signatures for fast cache validation."""
        sig: dict[str, Any] = {}
        for fname in ["raw_scans.zip", "refinement_pkls.zip", "ledger_precursor_genome.json"]:
            fpath = os.path.join(self.data_dir, fname)
            if os.path.exists(fpath):
                stat = os.stat(fpath)
                sig[fname] = {"size": stat.st_size, "mtime": stat.st_mtime}
        return sig

    def build_or_load(self, samples: list[dict] | None = None) -> ALabArtifactIndex:
        """Loads index from local cache if valid against source archives, or builds fresh."""
        if self._initialized:
            return self

        cache_file = os.path.join(self.cache_dir, "artifact_index.json")
        current_sig = self._get_archive_stat_signature()

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_sig = cached.get("_meta_signature")
                # Verify cache freshness
                if cached_sig == current_sig:
                    self._index = {
                        sid: {mod: ArtifactRef.from_dict(ref_data) for mod, ref_data in mod_dict.items()}
                        for sid, mod_dict in cached.items()
                        if sid != "_meta_signature"
                    }
                    self._initialized = True
                    logger.info("Loaded A-Lab artifact index from validated cache (%d candidates).", len(self._index))
                    return self
                else:
                    logger.info("A-Lab source archives modified. Invalidating artifact index cache.")
            except Exception as e:
                logger.warning("Failed to load cached A-Lab artifact index: %s. Rebuilding.", e)

        self._build_index(samples=samples)
        self._save_cache(cache_file, signature=current_sig)
        self._initialized = True
        return self

    def _build_index(self, samples: list[dict] | None = None) -> None:
        """Scans ZIP central directories to index artifact members by sample ID without extraction."""
        raw_scans_zip = os.path.join(self.data_dir, "raw_scans.zip")
        ref_zip = os.path.join(self.data_dir, "refinement_pkls.zip")
        manifest_checksums = self._get_manifest_checksums()

        xrd_members: dict[str, tuple[str, int]] = {}
        if os.path.exists(raw_scans_zip):
            with zipfile.ZipFile(raw_scans_zip, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir() and info.filename.endswith(".xrdml"):
                        base = os.path.basename(info.filename)
                        xrd_members[info.filename] = (info.filename, info.file_size)
                        xrd_members[base] = (info.filename, info.file_size)

        ref_members: dict[str, tuple[str, int]] = {}
        if os.path.exists(ref_zip):
            with zipfile.ZipFile(ref_zip, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir() and info.filename.endswith(".pkl"):
                        base = os.path.basename(info.filename)
                        ref_members[info.filename] = (info.filename, info.file_size)
                        ref_members[base] = (info.filename, info.file_size)

        if samples is None:
            ledger_file = os.path.join(self.data_dir, "ledger_precursor_genome.json")
            if os.path.exists(ledger_file):
                with open(ledger_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples = data.get("samples", [])
            else:
                samples = []

        from src.domains.alab.canonical import get_canonical_refinement_case, get_canonical_scan

        self._index.clear()
        for s in samples:
            sid = s.get("sample_id")
            if not sid:
                continue
            self._index[sid] = {}

            # 1. Match canonical XRD scan strictly (fail closed if canonical scan artifact missing)
            can_scan, can_scan_idx, scan_method = get_canonical_scan(s)
            if can_scan is not None:
                fn = can_scan.get("filename", "")
                base = os.path.basename(fn)
                if fn in xrd_members:
                    mem_path, sz = xrd_members[fn]
                    self._index[sid]["XRD"] = ArtifactRef(
                        archive_path=raw_scans_zip,
                        member_path=mem_path,
                        modality="XRD",
                        size_bytes=sz,
                        checksum=manifest_checksums.get("raw_scans.zip"),
                        selected_scan_index=can_scan_idx,
                        selection_method=scan_method,
                        is_canonical=True,
                    )
                elif base in xrd_members:
                    mem_path, sz = xrd_members[base]
                    self._index[sid]["XRD"] = ArtifactRef(
                        archive_path=raw_scans_zip,
                        member_path=mem_path,
                        modality="XRD",
                        size_bytes=sz,
                        checksum=manifest_checksums.get("raw_scans.zip"),
                        selected_scan_index=can_scan_idx,
                        selection_method=scan_method,
                        is_canonical=True,
                    )

            # 2. Match canonical Refinement PKL strictly (no cross-scan or cross-case fallback)
            can_case, can_case_idx, case_method = get_canonical_refinement_case(can_scan)
            if can_case is not None:
                pkl = can_case.get("pkl_path", "")
                base_pkl = os.path.basename(pkl)
                if pkl in ref_members:
                    mem_path, sz = ref_members[pkl]
                    self._index[sid]["REFINEMENT"] = ArtifactRef(
                        archive_path=ref_zip,
                        member_path=mem_path,
                        modality="REFINEMENT",
                        size_bytes=sz,
                        checksum=manifest_checksums.get("refinement_pkls.zip"),
                        selected_scan_index=can_scan_idx,
                        selected_case_index=can_case_idx,
                        selection_method=case_method,
                        is_canonical=True,
                    )
                elif base_pkl in ref_members:
                    mem_path, sz = ref_members[base_pkl]
                    self._index[sid]["REFINEMENT"] = ArtifactRef(
                        archive_path=ref_zip,
                        member_path=mem_path,
                        modality="REFINEMENT",
                        size_bytes=sz,
                        checksum=manifest_checksums.get("refinement_pkls.zip"),
                        selected_scan_index=can_scan_idx,
                        selected_case_index=can_case_idx,
                        selection_method=case_method,
                        is_canonical=True,
                    )

    def resolve_artifact(
        self,
        sample_id: str,
        modality: str,
        sample: Mapping[str, Any] | None = None,
    ) -> ResolvedArtifactSelection:
        """Resolves artifact selection guaranteeing strict provenance and byte identity alignment."""
        ref = self.get_artifact_ref(sample_id, modality)
        if ref is not None:
            return ResolvedArtifactSelection(
                sample_id=sample_id,
                requested_modality=modality,
                selected_scan_index=ref.selected_scan_index,
                selected_case_index=ref.selected_case_index,
                artifact_ref=ref,
                selection_method=ref.selection_method,
                is_canonical=ref.is_canonical,
            )

        # Fall closed: canonical artifact missing
        scan_idx = None
        case_idx = None
        method = "unresolvable"
        if sample:
            from src.domains.alab.canonical import get_canonical_refinement_case, get_canonical_scan

            can_scan, can_scan_idx, scan_m = get_canonical_scan(sample)
            scan_idx = can_scan_idx
            if modality == "XRD":
                method = f"canonical_scan_{scan_m}_missing_artifact"
            elif modality == "REFINEMENT":
                can_case, can_case_idx, case_m = get_canonical_refinement_case(can_scan)
                case_idx = can_case_idx
                method = f"canonical_case_{case_m}_missing_artifact"

        return ResolvedArtifactSelection(
            sample_id=sample_id,
            requested_modality=modality,
            selected_scan_index=scan_idx,
            selected_case_index=case_idx,
            artifact_ref=None,
            selection_method=method,
            is_canonical=True,
        )

    def _save_cache(self, cache_file: str, signature: dict[str, Any] | None = None) -> None:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            serializable: dict[str, Any] = {
                sid: {mod: ref.to_dict() for mod, ref in mod_dict.items()}
                for sid, mod_dict in self._index.items()
            }
            if signature:
                serializable["_meta_signature"] = signature
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning("Could not persist A-Lab artifact index cache: %s", e)

    def get_artifact_ref(self, sample_id: str, modality: str) -> ArtifactRef | None:
        """Returns the ArtifactRef for a sample and modality, if present."""
        return self._index.get(sample_id, {}).get(modality)

    def has_artifact(self, sample_id: str, modality: str) -> bool:
        """Returns True if an artifact exists for the given sample and modality."""
        return sample_id in self._index and modality in self._index[sample_id]

    def read_artifact_bytes(self, ref: ArtifactRef) -> bytes:
        """Reads artifact bytes directly from its enclosing archive."""
        if not os.path.exists(ref.archive_path):
            raise FileNotFoundError(f"Archive file not found: {ref.archive_path}")
        with zipfile.ZipFile(ref.archive_path, "r") as zf:
            return zf.read(ref.member_path)

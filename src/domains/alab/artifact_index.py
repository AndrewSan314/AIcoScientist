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

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, str | int | None]) -> ArtifactRef:
        return cls(
            archive_path=str(data["archive_path"]),
            member_path=str(data["member_path"]),
            modality=str(data["modality"]),
            size_bytes=int(data["size_bytes"]) if data.get("size_bytes") is not None else None,
            checksum=str(data["checksum"]) if data.get("checksum") is not None else None,
        )


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

    def build_or_load(self, samples: list[dict] | None = None) -> ALabArtifactIndex:
        """Loads index from local cache if valid, or builds from external archives."""
        if self._initialized:
            return self

        cache_file = os.path.join(self.cache_dir, "artifact_index.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                self._index = {
                    sid: {mod: ArtifactRef.from_dict(ref_data) for mod, ref_data in mod_dict.items()}
                    for sid, mod_dict in cached.items()
                }
                self._initialized = True
                logger.info("Loaded A-Lab artifact index from cache (%d candidates).", len(self._index))
                return self
            except Exception as e:
                logger.warning("Failed to load cached A-Lab artifact index: %s. Rebuilding.", e)

        self._build_index(samples=samples)
        self._save_cache(cache_file)
        self._initialized = True
        return self

    def _build_index(self, samples: list[dict] | None = None) -> None:
        """Scans ZIP central directories to index artifact members by sample ID without extraction."""
        raw_scans_zip = os.path.join(self.data_dir, "raw_scans.zip")
        ref_zip = os.path.join(self.data_dir, "refinement_pkls.zip")

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

        self._index.clear()
        for s in samples:
            sid = s.get("sample_id")
            if not sid:
                continue
            self._index[sid] = {}

            # Match XRD scan
            scans = s.get("characterization", {}).get("xrd", {}).get("scans", [])
            for sc in scans:
                fn = sc.get("filename", "")
                base = os.path.basename(fn)
                if fn in xrd_members:
                    mem_path, sz = xrd_members[fn]
                    self._index[sid]["XRD"] = ArtifactRef(
                        archive_path=raw_scans_zip,
                        member_path=mem_path,
                        modality="XRD",
                        size_bytes=sz,
                    )
                    break
                elif base in xrd_members:
                    mem_path, sz = xrd_members[base]
                    self._index[sid]["XRD"] = ArtifactRef(
                        archive_path=raw_scans_zip,
                        member_path=mem_path,
                        modality="XRD",
                        size_bytes=sz,
                    )
                    break

            # Match Refinement PKL
            for sc in scans:
                for rc in sc.get("refinement_cases", []):
                    pkl = rc.get("pkl_path", "")
                    base_pkl = os.path.basename(pkl)
                    if pkl in ref_members:
                        mem_path, sz = ref_members[pkl]
                        self._index[sid]["REFINEMENT"] = ArtifactRef(
                            archive_path=ref_zip,
                            member_path=mem_path,
                            modality="REFINEMENT",
                            size_bytes=sz,
                        )
                        break
                    elif base_pkl in ref_members:
                        mem_path, sz = ref_members[base_pkl]
                        self._index[sid]["REFINEMENT"] = ArtifactRef(
                            archive_path=ref_zip,
                            member_path=mem_path,
                            modality="REFINEMENT",
                            size_bytes=sz,
                        )
                        break
                if "REFINEMENT" in self._index[sid]:
                    break

    def _save_cache(self, cache_file: str) -> None:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            serializable = {
                sid: {mod: ref.to_dict() for mod, ref in mod_dict.items()}
                for sid, mod_dict in self._index.items()
            }
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

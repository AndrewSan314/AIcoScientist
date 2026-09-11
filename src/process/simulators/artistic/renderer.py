from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .config import ArtisticRunConfig
from .provenance import sha256 as _streaming_sha256


_PLACEHOLDER = re.compile(r"@([A-Za-z0-9_]+)@")
_STAGE_DIR = {"slurry": "Slurry", "drying_homogeneous": "Drying_homogeneous", "drying_heterogeneous": "Drying_heterogeneous", "calendering": "Calendering"}


@dataclass
class RenderState:
    workspace: Path
    source_hashes: dict[str, str] = field(default_factory=dict)
    patches: list[dict[str, object]] = field(default_factory=list)


class ArtisticRenderer:
    """Copies the pinned source into one isolated run workspace; never edits it."""

    def __init__(self, config: ArtisticRunConfig) -> None:
        self.config = config

    def prepare_workspace(self, run_id: str) -> RenderState:
        self.config.verify_source_pin()
        if not self.config.source_tree.is_dir():
            raise FileNotFoundError(f"Pinned ARTISTIC source tree is unavailable: {self.config.source_tree}")
        run_directory = self.config.output_root / run_id
        if run_directory.exists():
            raise FileExistsError(f"Refusing to append to ARTISTIC run directory: {run_directory}")
        workspace = run_directory / "workspace"
        workspace.mkdir(parents=True)
        return RenderState(workspace=workspace)

    def stage(self, state: RenderState, name: str, values: Mapping[str, float | int]) -> None:
        source = self.config.source_tree / _STAGE_DIR[name]
        if not source.is_dir():
            raise FileNotFoundError(f"ARTISTIC stage directory missing: {source}")
        for source_path in source.iterdir():
            destination = state.workspace / source_path.name
            if source_path.is_dir():
                shutil.copytree(source_path, destination, dirs_exist_ok=True)
            else:
                state.source_hashes[f"NMC/Updated version/{_STAGE_DIR[name]}/{source_path.name}"] = _sha256(source_path)
                shutil.copy2(source_path, destination)
        if name == "slurry":
            if self.config.apply_verified_patches:
                self._patch_am_fraction_indices(state)
            self._patch_fidelity(state)
        for path in state.workspace.glob("user_inputs*.txt"):
            self._substitute(path, values)

    @staticmethod
    def _substitute(path: Path, values: Mapping[str, float | int]) -> None:
        text = path.read_text(encoding="utf-8")
        missing = sorted(set(_PLACEHOLDER.findall(text)) - set(values))
        if missing:
            raise ValueError(f"{path.name} contains unbound ARTISTIC controls: {', '.join(missing)}")
        rendered = _PLACEHOLDER.sub(lambda match: _format(values[match.group(1)]), text)
        if _PLACEHOLDER.search(rendered):
            raise ValueError(f"unresolved ARTISTIC placeholder remains in {path.name}")
        path.write_text(rendered, encoding="utf-8", newline="\n")

    @staticmethod
    def _patch_am_fraction_indices(state: RenderState) -> None:
        path = state.workspace / "init_structure.txt"
        text = path.read_text(encoding="utf-8")
        original = text
        for index in range(7, 11):
            old = f"variable n_AM{index} equal round(v_n_AM*v_p_AM6)"
            new = f"variable n_AM{index} equal round(v_n_AM*v_p_AM{index})"
            if old not in text:
                raise ValueError(f"verified ARTISTIC patch target missing: {old}")
            text = text.replace(old, new, 1)
        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            state.patches.append({
                "id": "slurry_am_fraction_indices", "path": "Slurry/init_structure.txt",
                "before_sha256": hashlib.sha256(original.encode()).hexdigest(), "after_sha256": _sha256(path),
                "reason": "upstream n_AM7..10 each used p_AM6 instead of their corresponding particle fraction",
            })

    def _patch_fidelity(self, state: RenderState) -> None:
        if self.config.fidelity_mode.value != "SHORT_HORIZON":
            return
        path = state.workspace / "in_slurry.run"
        text = path.read_text(encoding="utf-8")
        pattern = re.compile(r"(?m)^(\s*variable\s+run\s+equal\s+)20000000(\s*(?:#.*)?)$")
        match = pattern.search(text)
        if not match:
            raise ValueError("short-horizon ARTISTIC patch target missing: variable run equal 20000000")
        replacement = f"{match.group(1)}{self.config.requested_slurry_steps}{match.group(2)}"
        rendered = text[:match.start()] + replacement + text[match.end():]
        path.write_text(rendered, encoding="utf-8", newline="\n")
        state.patches.append({
            "id": "short_horizon_slurry_steps", "path": "Slurry/in_slurry.run",
            "original": match.group(0), "new": replacement,
            "before_sha256": hashlib.sha256(text.encode()).hexdigest(), "after_sha256": _sha256(path),
            "requested_steps": self.config.requested_slurry_steps,
        })
        dump_pattern = re.compile(r"(?m)^(\s*dump\s+\S+\s+.*?\bcustom\s+)1000000(\b.*)$")
        dump_match = dump_pattern.search(rendered)
        if dump_match and self.config.dump_interval_steps != 1_000_000:
            dump_replacement = f"{dump_match.group(1)}{self.config.dump_interval_steps}{dump_match.group(2)}"
            before_dump = rendered
            rendered = rendered[:dump_match.start()] + dump_replacement + rendered[dump_match.end():]
            path.write_text(rendered, encoding="utf-8", newline="\n")
            state.patches.append({
                "id": "short_horizon_checkpoint_interval", "path": "Slurry/in_slurry.run",
                "original": dump_match.group(0), "new": dump_replacement,
                "before_sha256": hashlib.sha256(before_dump.encode()).hexdigest(), "after_sha256": _sha256(path),
                "requested_interval_steps": self.config.dump_interval_steps,
            })


def _format(value: float | int) -> str:
    return str(value) if isinstance(value, int) else format(float(value), ".12g")


def _sha256(path: Path) -> str:
    return _streaming_sha256(path)

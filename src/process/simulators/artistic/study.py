from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from . import config as config_module
from .config import ArtisticRunConfig, FidelityMode, REFERENCE_SLURRY_STEPS, physics_config_fingerprint
from .schemas import ArtisticRecipe, estimate_particles, recipe_fingerprint


DEFAULT_STUDY_HORIZONS = (500_000, 1_000_000, 2_000_000)


@dataclass(frozen=True)
class StudyPlan:
    horizons: tuple[int, ...]
    dump_interval_steps: int
    dry_run: bool
    entries: tuple[dict[str, object], ...]


def build_convergence_study_plan(
    config: ArtisticRunConfig,
    *,
    horizons: Iterable[int] = DEFAULT_STUDY_HORIZONS,
    steps_per_second: float | None = None,
    dry_run: bool = True,
    recipe: ArtisticRecipe | None = None,
) -> StudyPlan:
    selected = tuple(dict.fromkeys(int(step) for step in horizons))
    if not selected or any(step <= 0 or step > REFERENCE_SLURRY_STEPS for step in selected):
        raise ValueError(f"study horizons must be positive and no greater than {REFERENCE_SLURRY_STEPS:,}")
    if steps_per_second is not None and steps_per_second <= 0:
        raise ValueError("steps_per_second must be positive when supplied")
    entries: list[dict[str, object]] = []
    particles = estimate_particles(recipe).as_dict() if recipe is not None else None
    recipe_id = recipe_fingerprint(recipe) if recipe is not None else None
    pinned_commit = config_module.PINNED_COMMIT
    pinned_source_tree_hash = config_module.PINNED_SOURCE_TREE_HASH
    physics_id = physics_config_fingerprint(
        recipe_fingerprint=recipe_id, source_commit=pinned_commit, source_tree_hash=pinned_source_tree_hash,
        patches=config_module.VERIFIED_PHYSICS_PATCHES if config.apply_verified_patches else (),
    ) if recipe_id is not None else None
    for steps in selected:
        is_reference = steps == REFERENCE_SLURRY_STEPS
        mode = FidelityMode.REFERENCE if is_reference else FidelityMode.SHORT_HORIZON
        entries.append({
            "fidelity_mode": mode.value,
            "fidelity_identity": _fidelity_identity(mode, steps, config.dump_interval_steps),
            "recipe_fingerprint": recipe_id,
            "pinned_commit": pinned_commit,
            "pinned_source_tree_hash": pinned_source_tree_hash,
            "physics_config_fingerprint": physics_id,
            "particle_preflight": particles,
            "execution_mode": config.execution_mode.value,
            "mpi_processes": config.mpi_processes,
            "requested_dynamic_steps": steps,
            "requested_slurry_steps": steps,
            "dump_interval_steps": config.dump_interval_steps,
            "estimated_wall_seconds": steps / steps_per_second if steps_per_second else None,
            "requires_explicit_reference_confirmation": is_reference,
            "auto_launch": False,
        })
    return StudyPlan(selected, config.dump_interval_steps, dry_run, tuple(entries))


def _fidelity_identity(mode: FidelityMode, steps: int, dump_interval_steps: int) -> str:
    payload = {
        "mode": mode.value, "reference_slurry_steps": REFERENCE_SLURRY_STEPS,
        "requested_slurry_steps": steps, "dump_interval_steps": dump_interval_steps,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

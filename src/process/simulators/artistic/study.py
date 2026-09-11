from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import ArtisticRunConfig, FidelityMode, REFERENCE_SLURRY_STEPS


DEFAULT_STUDY_HORIZONS = (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, REFERENCE_SLURRY_STEPS)


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
) -> StudyPlan:
    selected = tuple(dict.fromkeys(int(step) for step in horizons))
    if not selected or any(step <= 0 or step > REFERENCE_SLURRY_STEPS for step in selected):
        raise ValueError(f"study horizons must be positive and no greater than {REFERENCE_SLURRY_STEPS:,}")
    if steps_per_second is not None and steps_per_second <= 0:
        raise ValueError("steps_per_second must be positive when supplied")
    entries: list[dict[str, object]] = []
    for steps in selected:
        is_reference = steps == REFERENCE_SLURRY_STEPS
        entries.append({
            "fidelity_mode": FidelityMode.REFERENCE.value if is_reference else FidelityMode.SHORT_HORIZON.value,
            "requested_slurry_steps": steps,
            "dump_interval_steps": config.dump_interval_steps,
            "estimated_wall_seconds": steps / steps_per_second if steps_per_second else None,
            "requires_explicit_reference_confirmation": is_reference,
            "auto_launch": False,
        })
    return StudyPlan(selected, config.dump_interval_steps, dry_run, tuple(entries))

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


_ARTISTIC_PI = 3.1415926

class DryingMode(StrEnum):
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _fraction(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class SlurryRecipe:
    n_am_part: int
    diameter_am_um: tuple[float, ...]
    percent_am: tuple[float, ...]
    diameter_cbd_solid_um: float
    solid_content: float
    electrode_mass_ug: float
    ratio_am: float
    ratio_cbd: float
    thickness_level: int
    cbd_nanoporosity: float

    def __post_init__(self) -> None:
        if not 1 <= self.n_am_part <= 10:
            raise ValueError("n_am_part must be in [1, 10], as required by user_inputs.txt")
        if len(self.diameter_am_um) != 10 or len(self.percent_am) != 10:
            raise ValueError("ARTISTIC user_inputs.txt requires exactly ten AM diameter and fraction values")
        if any(_finite("diameter_am_um", value) <= 0 for value in self.diameter_am_um):
            raise ValueError("diameter_am_um must be positive")
        if any(_fraction("percent_am", value) < 0 for value in self.percent_am):
            raise ValueError("percent_am must be non-negative")
        if abs(sum(self.percent_am[: self.n_am_part]) - 1) > 1e-9:
            raise ValueError("active ARTISTIC AM particle fractions must sum to one")
        if any(value != 0 for value in self.percent_am[self.n_am_part :]):
            raise ValueError("inactive ARTISTIC AM particle fractions must be zero")
        if not 0.7 <= _finite("diameter_cbd_solid_um", self.diameter_cbd_solid_um) <= 1.5:
            raise ValueError("diameter_cbd_solid_um must be in the published ARTISTIC range [0.7, 1.5] um")
        if not 0.42 <= _fraction("solid_content", self.solid_content) <= 0.70:
            raise ValueError("solid_content must be in the published ARTISTIC range [0.42, 0.70]")
        if not 0.1 <= _finite("electrode_mass_ug", self.electrode_mass_ug) <= 0.2:
            raise ValueError("electrode_mass_ug must be in the published ARTISTIC range [0.1, 0.2] ug")
        _fraction("ratio_am", self.ratio_am)
        _fraction("ratio_cbd", self.ratio_cbd)
        if abs(self.ratio_am + self.ratio_cbd - 1) > 1e-9:
            raise ValueError("ratio_am + ratio_cbd must equal one")
        if not 0.85 <= self.ratio_am <= 0.97 or not 0.03 <= self.ratio_cbd <= 0.15:
            raise ValueError("AM/CBD formulation must be in the published ARTISTIC range")
        if self.thickness_level not in (0, 1):
            raise ValueError("thickness_level must be 0 or 1")
        if not 0.3 <= _fraction("cbd_nanoporosity", self.cbd_nanoporosity) <= 0.7:
            raise ValueError("cbd_nanoporosity must be in the published ARTISTIC range [0.3, 0.7]")
        if any(not 2 <= value <= 25 for value in self.diameter_am_um[: self.n_am_part]):
            raise ValueError("active diameter_am_um values must be in the published ARTISTIC range [2, 25] um")

    def template_values(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {"nAM_part": self.n_am_part}
        values.update({f"diameterAM{i + 1}": value for i, value in enumerate(self.diameter_am_um)})
        values.update({f"percentAM{i + 1}": value for i, value in enumerate(self.percent_am)})
        values.update({
            "diameterCB_solid": self.diameter_cbd_solid_um, "stl_ratio": self.solid_content,
            # Upstream user_inputs.txt converts this microgram input to grams with /1E6.
            "dry_mass": self.electrode_mass_ug, "ratio_AM": self.ratio_am, "ratio_CBD": self.ratio_cbd,
            "thickness": self.thickness_level, "CBD_nano": self.cbd_nanoporosity,
        })
        return values


@dataclass(frozen=True)
class HeterogeneousDryingRecipe:
    n_zones: int
    evaporation_mode: float
    evaporation_rate2: float
    evaporation_rate3: float = 1.0

    def __post_init__(self) -> None:
        if self.n_zones not in (2, 3):
            raise ValueError("n_zones must be 2 or 3 according to user_inputs_evHet.txt")
        if self.evaporation_mode not in (1, 0.5, 0.33, 2, 3):
            raise ValueError("evaporation_mode must be one of ARTISTIC's documented modes")
        if _finite("evaporation_rate2", self.evaporation_rate2) <= 0 or _finite("evaporation_rate3", self.evaporation_rate3) <= 0:
            raise ValueError("ARTISTIC evaporation rates must be positive")
        if self.n_zones == 2 and self.evaporation_rate3 != 1:
            raise ValueError("two-zone drying fixes evaporation_rate3 to 1 in the source")

    def template_values(self) -> dict[str, float | int]:
        return {"nzones": self.n_zones, "evap_mode": self.evaporation_mode, "evap_rate2": self.evaporation_rate2, "evap_rate3": self.evaporation_rate3}


@dataclass(frozen=True)
class CalenderingRecipe:
    compression_degree: float
    cbd_nanoporosity_decrease: float
    relaxation: bool
    perform_energy_minimization: bool

    def __post_init__(self) -> None:
        if not 0.05 <= _fraction("compression_degree", self.compression_degree) <= 0.40:
            raise ValueError("compression_degree must be in the published ARTISTIC range [0.05, 0.40]")
        _fraction("cbd_nanoporosity_decrease", self.cbd_nanoporosity_decrease)

    def template_values(self) -> dict[str, float | int]:
        # Source code executes `minimize` when its `minimize` variable equals 1.
        return {"cal_degree": self.compression_degree, "nano_dec": self.cbd_nanoporosity_decrease, "relax": int(self.relaxation), "minimize": int(self.perform_energy_minimization)}


@dataclass(frozen=True)
class ArtisticRecipe:
    slurry: SlurryRecipe
    drying_mode: DryingMode | None = None
    heterogeneous_drying: HeterogeneousDryingRecipe | None = None
    calendering: CalenderingRecipe | None = None

    def __post_init__(self) -> None:
        if self.heterogeneous_drying and self.drying_mode != DryingMode.HETEROGENEOUS:
            raise ValueError("heterogeneous_drying requires drying_mode='heterogeneous'")
        if self.drying_mode == DryingMode.HETEROGENEOUS and not self.heterogeneous_drying:
            raise ValueError("heterogeneous drying requires its source-controlled recipe")
        if self.calendering and not self.drying_mode:
            raise ValueError("calendering requires a completed drying stage")

    @property
    def template_values(self) -> Mapping[str, float | int]:
        values = dict(self.slurry.template_values())
        if self.heterogeneous_drying:
            values.update(self.heterogeneous_drying.template_values())
        if self.calendering:
            values.update(self.calendering.template_values())
        return values


class ParticleCountSafetyError(RuntimeError):
    """A recipe exceeds the local simulator safety budget before execution."""


@dataclass(frozen=True)
class ParticleEstimate:
    n_am_nominal: int
    n_am_by_type: tuple[int, ...]
    n_am_created_total: int
    n_cbd: int

    @property
    def n_am(self) -> int:
        """Backward-compatible alias for the nominal AM count."""
        return self.n_am_nominal

    @property
    def total_particles(self) -> int:
        return self.n_am_created_total + self.n_cbd

    @property
    def estimated_memory_bytes(self) -> int:
        return self.total_particles * 1_600

    def as_dict(self) -> dict[str, object]:
        gib = self.estimated_memory_bytes / 1024**3
        return {
            "n_AM": self.n_am_nominal,
            "n_AM_nominal": self.n_am_nominal,
            "n_AM_by_type": list(self.n_am_by_type),
            "n_AM_created_total": self.n_am_created_total,
            "n_CBD": self.n_cbd,
            "total_particles": self.total_particles,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "estimated_memory_gib": gib,
            "memory_class": "< 1 GiB" if gib < 1 else "1-16 GiB" if gib < 16 else ">= 16 GiB",
            "memory_basis": "heuristic 1,600 bytes/particle estimate; observed peak can be higher",
        }


def _lammps_round(value: float) -> int:
    """Match LAMMPS/C round semantics for the positive particle counts here."""
    return math.floor(value + 0.5)


def estimate_particles(recipe: ArtisticRecipe) -> ParticleEstimate:
    """Mirror ARTISTIC's mass/density particle-count arithmetic before LAMMPS runs."""
    slurry = recipe.slurry
    mass_g = slurry.electrode_mass_ug / 1e6  # exact upstream `dry_mass` conversion
    am_volume_um3 = mass_g * slurry.ratio_am / 4.65 * 1e12
    mean_am_volume_um3 = sum(
        fraction * _ARTISTIC_PI * diameter**3 / 6
        for diameter, fraction in zip(slurry.diameter_am_um[: slurry.n_am_part], slurry.percent_am[: slurry.n_am_part])
    )
    cbd_volume_um3 = mass_g * slurry.ratio_cbd / (1.8 * slurry.cbd_nanoporosity) * 1e12
    cbd_particle_volume_um3 = _ARTISTIC_PI * slurry.diameter_cbd_solid_um**3 / 6
    n_am_nominal = _lammps_round(am_volume_um3 / mean_am_volume_um3)
    n_am_by_type = tuple(_lammps_round(n_am_nominal * fraction) for fraction in slurry.percent_am[: slurry.n_am_part])
    return ParticleEstimate(
        n_am_nominal=n_am_nominal,
        n_am_by_type=n_am_by_type,
        n_am_created_total=sum(n_am_by_type),
        n_cbd=_lammps_round(cbd_volume_um3 / cbd_particle_volume_um3),
    )


def recipe_fingerprint(recipe: ArtisticRecipe) -> str:
    """Stable group identity for the physical controls, never a simulator execution."""
    payload = {
        "slurry": recipe.slurry.template_values(),
        "drying_mode": recipe.drying_mode.value if recipe.drying_mode else None,
        "heterogeneous_drying": recipe.heterogeneous_drying.template_values() if recipe.heterogeneous_drying else None,
        "calendering": recipe.calendering.template_values() if recipe.calendering else None,
    }
    encoded = json.dumps(_canonical_controls(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_controls(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical_controls(item) for key, item in value.items()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return "0" if number == 0 else format(number, ".17g")
    return value

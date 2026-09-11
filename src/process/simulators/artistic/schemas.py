from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


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
    dry_mass_mg: float
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
        if _finite("diameter_cbd_solid_um", self.diameter_cbd_solid_um) <= 0:
            raise ValueError("diameter_cbd_solid_um must be positive")
        if not 0 < _fraction("solid_content", self.solid_content) <= 1:
            raise ValueError("solid_content must be in (0, 1]")
        if _finite("dry_mass_mg", self.dry_mass_mg) <= 0:
            raise ValueError("dry_mass_mg must be positive")
        _fraction("ratio_am", self.ratio_am)
        _fraction("ratio_cbd", self.ratio_cbd)
        if abs(self.ratio_am + self.ratio_cbd - 1) > 1e-9:
            raise ValueError("ratio_am + ratio_cbd must equal one")
        if self.thickness_level not in (0, 1):
            raise ValueError("thickness_level must be 0 or 1")
        _fraction("cbd_nanoporosity", self.cbd_nanoporosity)

    def template_values(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {"nAM_part": self.n_am_part}
        values.update({f"diameterAM{i + 1}": value for i, value in enumerate(self.diameter_am_um)})
        values.update({f"percentAM{i + 1}": value for i, value in enumerate(self.percent_am)})
        values.update({
            "diameterCB_solid": self.diameter_cbd_solid_um, "stl_ratio": self.solid_content,
            "dry_mass": self.dry_mass_mg, "ratio_AM": self.ratio_am, "ratio_CBD": self.ratio_cbd,
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
        _fraction("compression_degree", self.compression_degree)
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

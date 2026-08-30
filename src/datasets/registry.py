from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import DatasetAdapter

AVAILABLE_DATASETS = (
    "si_mxene",
    "severson",
    "dynamic_cycling",
    "attia",
    "feconi",
    "feconi_kerr",
    "feconi_coercivity",
    "auirh",
    "auirh_k0",
    "auirh_i_lim",
    "auirh_alpha",
)


def get_dataset_adapter(name: str) -> DatasetAdapter:
    """Returns the instantiated dataset adapter lazily to avoid unnecessary dependency imports."""
    dataset_key = name.lower().strip()
    if dataset_key == "si_mxene":
        from .si_mxene import SiMxeneAdapter
        return SiMxeneAdapter()
    elif dataset_key == "severson":
        from .severson import SeversonAdapter
        return SeversonAdapter()
    elif dataset_key == "dynamic_cycling":
        from .dynamic_cycling import DynamicCyclingAdapter
        return DynamicCyclingAdapter()
    elif dataset_key == "attia":
        from .attia import AttiaAdapter
        return AttiaAdapter()
    elif dataset_key in {"feconi", "feconi_kerr"}:
        from .feconi import FeCoNiAdapter
        return FeCoNiAdapter(target="Kerr")
    elif dataset_key in {"feconi_coer", "feconi_coercivity"}:
        from .feconi import FeCoNiAdapter
        return FeCoNiAdapter(target="Coer")
    elif dataset_key in {"auirh", "auirh_k0"}:
        from .auirh import AuIrRhAdapter
        return AuIrRhAdapter(target="k0")
    elif dataset_key in {"auirh_i_lim", "auirh_ilim"}:
        from .auirh import AuIrRhAdapter
        return AuIrRhAdapter(target="i_lim")
    elif dataset_key == "auirh_alpha":
        from .auirh import AuIrRhAdapter
        return AuIrRhAdapter(target="alpha")
    else:
        available = ", ".join(sorted(AVAILABLE_DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {available}")

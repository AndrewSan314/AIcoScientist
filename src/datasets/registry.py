from types import MappingProxyType

from .attia import AttiaAdapter
from .base import DatasetAdapter
from .dynamic_cycling import DynamicCyclingAdapter
from .severson import SeversonAdapter
from .si_mxene import SiMxeneAdapter

DATASET_ADAPTERS = MappingProxyType(
    {
        "si_mxene": SiMxeneAdapter,
        "severson": SeversonAdapter,
        "dynamic_cycling": DynamicCyclingAdapter,
        "attia": AttiaAdapter,
    }
)


def get_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        adapter_type = DATASET_ADAPTERS[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_ADAPTERS))
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {available}") from error
    return adapter_type()


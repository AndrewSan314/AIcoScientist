from .base import DatasetAdapter, DatasetSpec


def get_dataset_adapter(name: str) -> DatasetAdapter:
    from .registry import get_dataset_adapter as get_registered_adapter

    return get_registered_adapter(name)

__all__ = ["DatasetAdapter", "DatasetSpec", "get_dataset_adapter"]

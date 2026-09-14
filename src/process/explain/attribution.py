from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.process.dependency_graph import ProcessDependencyGraph
from src.process.models.flat_baseline import TreeEnsembleBaseline


@dataclass(frozen=True)
class Attribution:
    name: str
    contribution: float
    method: str
    interpretation: str = "predictive model association; not causal effect"


def tree_feature_attributions(model: TreeEnsembleBaseline, feature_names: Sequence[str]) -> tuple[Attribution, ...]:
    if not hasattr(model.model, "feature_importances_"):
        raise ValueError("fit a tree ensemble before requesting feature attribution")
    if len(feature_names) != len(model.model.feature_importances_) or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must uniquely match the fitted tree input dimension")
    return tuple(sorted((Attribution(str(name), float(value), "tree_impurity_importance") for name, value in zip(feature_names, model.model.feature_importances_, strict=True)), key=lambda item: (-item.contribution, item.name)))


def ablation_delta(reference: float, ablated: float, *, name: str) -> Attribution:
    if not name.strip() or not np.isfinite([reference, ablated]).all():
        raise ValueError("ablation requires a name and finite metrics")
    return Attribution(name, float(ablated - reference), "held_out_ablation_delta", "held-out predictive metric delta; not causal effect")


def graph_path_attribution(graph: ProcessDependencyGraph, source_node: str, target_node: str) -> tuple[str, ...]:
    """Return the domain-informed graph path used by the process architecture prior."""
    return graph.path_attribution(source_node, target_node)

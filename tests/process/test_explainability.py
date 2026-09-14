from __future__ import annotations

import numpy as np
import pytest

from src.process.dependency_graph import ProcessDependencyGraph
from src.process.explain import ablation_delta, graph_path_attribution, tree_feature_attributions
from src.process.models.flat_baseline import TreeEnsembleBaseline
from src.process.stages import ProcessStage


def test_tree_attribution_and_ablation_are_predictive_only() -> None:
    model = TreeEnsembleBaseline(n_estimators=8).fit(np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), np.array([0.0, 1.0, 2.0]))
    attributions = tree_feature_attributions(model, ("speed", "constant"))
    assert attributions[0].name == "speed"
    assert "not causal" in ablation_delta(0.9, 0.7, name="no_ultrasound").interpretation


def test_graph_path_attribution_is_the_declared_architecture_prior() -> None:
    graph = ProcessDependencyGraph(("mix", "coat"), (("mix", "coat"),), {"mix": ProcessStage.MIXING, "coat": ProcessStage.COATING})
    assert graph_path_attribution(graph, "mix", "coat") == ("mix", "coat")
    with pytest.raises(ValueError):
        ablation_delta(1.0, float("nan"), name="bad")

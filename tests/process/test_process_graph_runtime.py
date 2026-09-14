from pathlib import Path

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.dependency_graph import ProcessDependencyGraph
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.stages import ProcessStage
from .test_maspo_control import FirstCandidateBackend, _horizon, _observations, _space


def test_default_graph_exposes_legal_paths_and_stage_transitions():
    graph = ProcessDependencyGraph.from_yaml("config/process_graphs/li_ion_electrode.yaml")
    assert ProcessStage.DRYING in graph.legal_successor_stages(ProcessStage.COATING)
    graph.validate_transition(ProcessStage.COATING, ProcessStage.DRYING)
    assert graph.path_attribution("formulation", "capacity")[-1] == "capacity"
    assert graph.provenance()["stage_order_source"] == "PROCESS_DEPENDENCY_GRAPH"


def test_maspo_uses_graph_for_next_legal_control_stage():
    plan = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=FirstCandidateBackend())).optimize_remaining_process(
        current_state=_horizon(0.8), current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: _space(), ProcessStage.CALENDERING: _space()}, observations=_observations(),
        objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
    )
    assert plan.next_stage == ProcessStage.DRYING
    assert plan.graph_provenance["stage_order_source"] == "PROCESS_DEPENDENCY_GRAPH"


def test_maspo_records_explicit_stage_order_fallback(tmp_path: Path):
    plan = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=FirstCandidateBackend()), graph_path=tmp_path / "missing.yaml").optimize_remaining_process(
        current_state=_horizon(0.8), current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: _space()}, observations=_observations(),
        objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
    )
    assert plan.graph_provenance["stage_order_source"] == "STAGE_ORDER_FALLBACK"

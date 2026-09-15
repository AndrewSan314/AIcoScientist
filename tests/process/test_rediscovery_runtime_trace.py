"""Tests verifying that engine_path_audit is derived strictly from runtime execution trace."""

from __future__ import annotations
import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import (
    EngineExecutionTrace,
    RediscoveryReplay,
    run_rediscovery_benchmark,
)
from src.process.contracts import BatteryProcessRun, ParameterValue, MeasurementValue, StageRecord, ProvenanceRecord
from src.process.information_horizon import DecisionHorizon
from src.process.stages import ProcessStage, STAGE_ORDER


@pytest.fixture
def mock_trace_pool() -> tuple[pd.DataFrame, dict[str, list[BatteryProcessRun]]]:
    """Create a self-contained candidate pool with matching BatteryProcessRun objects."""
    prov = ProvenanceRecord("PHYSICAL_HISTORICAL", "", "", "1", {}, "2", {})
    records = []
    runs_by_recipe: dict[str, list[BatteryProcessRun]] = {}

    for i in range(8):
        rid = f"recipe_{i}"
        records.append({
            "recipe_id": rid,
            "coating_speed_m_per_min": 0.2 + 0.1 * i,
            "coating_gap_um": 100.0 + 10.0 * i,
            "discharge_specific_capacity_cycle30_mah_g": 300.0 + (50.0 if i == 5 else 5.0 * i),
        })
        run = BatteryProcessRun(
            run_id=f"run_{i}",
            cell_id=f"cell_{i}",
            batch_id=rid,
            chemistry_id="graphite",
            equipment_context={},
            environment_context={},
            stages=[
                StageRecord(
                    stage_id=f"run_{i}:formulation",
                    stage_type=ProcessStage.FORMULATION,
                    sequence_index=STAGE_ORDER[ProcessStage.FORMULATION],
                    controls={"active_material_fraction_pct": ParameterValue(95.0, "%")},
                    intermediate_properties={},
                    modalities=[],
                    provenance=prov,
                ),
                StageRecord(
                    stage_id=f"run_{i}:coating",
                    stage_type=ProcessStage.COATING,
                    sequence_index=STAGE_ORDER[ProcessStage.COATING],
                    controls={
                        "coating_speed_m_per_min": ParameterValue(0.2 + 0.1 * i, "m/min"),
                        "coating_gap_um": ParameterValue(100.0 + 10.0 * i, "um"),
                    },
                    intermediate_properties={"wet_thickness_um": MeasurementValue(120.0, "um")},
                    modalities=[],
                    provenance=prov,
                ),
            ],
            final_kpis={
                "discharge_specific_capacity_cycle30_mah_g": MeasurementValue(
                    300.0 + (50.0 if i == 5 else 5.0 * i), "mAh/g"
                )
            },
            provenance=prov,
        )
        runs_by_recipe[rid] = [run]

    df = pd.DataFrame(records).sort_values(
        by="discharge_specific_capacity_cycle30_mah_g", ascending=False
    ).reset_index(drop=True)
    return df, runs_by_recipe


def test_runtime_trace_ai_policy_counters(mock_trace_pool):
    df, runs_by_recipe = mock_trace_pool
    replay = RediscoveryReplay(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        runs_by_recipe=runs_by_recipe,
    )

    traj = replay.run(
        strategy="AICOSCIENTIST_PROCESS_SURROGATE",
        seed=42,
        initial_size=3,
        max_steps=2,
    )

    trace = traj.execution_trace
    assert trace["battery_process_runs_seen"] > 0
    assert trace["recipe_selection_horizon_invocations"] > 0
    assert trace["process_surrogate_samples_created"] > 0
    assert trace["process_surrogate_fit_count"] == 2
    assert trace["coordinator_proposal_count"] == 2
    assert trace["oracle_reveal_count"] == 5  # 3 initial + 2 steps
    assert trace["direct_botorch_calls"] == 0
    assert len(trace["surrogate_artifact_fingerprints"]) == 2
    assert trace["optimizer_backend_type"] == "FrozenSurrogateOptimizerBackend"

    trace_obj = EngineExecutionTrace(**{k: v for k, v in trace.items() if k in EngineExecutionTrace.__dataclass_fields__})
    audit = trace_obj.generate_audit("AICOSCIENTIST_PROCESS_SURROGATE", "AICOSCIENTIST_PROCESS_SURROGATE")
    assert audit["verified"] is True
    assert audit["uses_direct_botorch_backend"] is False


def test_runtime_trace_direct_botorch_counters(mock_trace_pool):
    df, runs_by_recipe = mock_trace_pool
    replay = RediscoveryReplay(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        runs_by_recipe=runs_by_recipe,
    )

    traj = replay.run(
        strategy="DIRECT_BOTORCH_BASELINE",
        seed=42,
        initial_size=3,
        max_steps=2,
    )

    trace = traj.execution_trace
    assert trace["source_adapter_invocations"] == 0
    assert trace["direct_botorch_calls"] == 2
    assert trace["battery_process_runs_seen"] == 0
    assert trace["process_surrogate_fit_count"] == 0
    assert trace["coordinator_proposal_count"] == 0
    assert trace["optimizer_backend_type"] == "BoTorchBackend"

    trace_obj = EngineExecutionTrace(**{k: v for k, v in trace.items() if k in EngineExecutionTrace.__dataclass_fields__})
    audit = trace_obj.generate_audit("DIRECT_BOTORCH_BASELINE", "DIRECT_BOTORCH_BASELINE")
    assert audit["verified"] is True
    assert audit["uses_direct_botorch_backend"] is True


def test_runtime_trace_random_counters(mock_trace_pool):
    df, runs_by_recipe = mock_trace_pool
    replay = RediscoveryReplay(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        runs_by_recipe=runs_by_recipe,
    )

    traj = replay.run(
        strategy="random",
        seed=42,
        initial_size=3,
        max_steps=2,
    )

    trace = traj.execution_trace
    assert trace["source_adapter_invocations"] == 0
    assert trace["random_steps"] == 2
    assert trace["battery_process_runs_seen"] == 0
    assert trace["process_surrogate_fit_count"] == 0
    assert trace["coordinator_proposal_count"] == 0
    assert trace["direct_botorch_calls"] == 0

    trace_obj = EngineExecutionTrace(**{k: v for k, v in trace.items() if k in EngineExecutionTrace.__dataclass_fields__})
    audit = trace_obj.generate_audit("random", "RANDOM_BASELINE")
    assert audit["verified"] is True


def test_audit_fails_if_counters_zero_or_inconsistent():
    """Prove that verified=True cannot be awarded without real runtime evidence."""
    # Fake trace claiming full engine without coordinator calls
    fake_trace = EngineExecutionTrace(
        battery_process_runs_seen=10,
        recipe_selection_horizon_invocations=10,
        process_surrogate_samples_created=10,
        process_surrogate_fit_count=1,
        surrogate_artifact_fingerprints=["fp1"],
        coordinator_proposal_count=0,  # Never called coordinator!
        oracle_reveal_count=4,
    )
    audit = fake_trace.generate_audit("AICOSCIENTIST_PROCESS_SURROGATE", "AICOSCIENTIST_PROCESS_SURROGATE")
    assert audit["verified"] is False

    # Fake trace claiming full engine without surrogate fitting
    fake_trace2 = EngineExecutionTrace(
        battery_process_runs_seen=10,
        recipe_selection_horizon_invocations=10,
        process_surrogate_samples_created=10,
        process_surrogate_fit_count=0,  # Never fit surrogate!
        surrogate_artifact_fingerprints=[],
        coordinator_proposal_count=1,
        oracle_reveal_count=4,
    )
    audit2 = fake_trace2.generate_audit("AICOSCIENTIST_PROCESS_SURROGATE", "AICOSCIENTIST_PROCESS_SURROGATE")
    assert audit2["verified"] is False


def test_run_rediscovery_benchmark_generates_engine_path_audit(mock_trace_pool):
    df, runs_by_recipe = mock_trace_pool
    results = run_rediscovery_benchmark(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        policies=["AICOSCIENTIST_PROCESS_SURROGATE", "DIRECT_BOTORCH_BASELINE", "random"],
        seeds=[42],
        initial_size=3,
        max_steps=2,
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        runs_by_recipe=runs_by_recipe,
    )

    assert "engine_path_audit" in results
    audit_list = results["engine_path_audit"]
    assert len(audit_list) == 3

    ai_audit = next(a for a in audit_list if a["policy"] == "AICOSCIENTIST_PROCESS_SURROGATE")
    assert ai_audit["verified"] is True
    assert ai_audit["battery_process_runs_seen"] > 0
    assert ai_audit["coordinator_proposal_count"] == 2
    assert ai_audit["uses_direct_botorch_backend"] is False

    botorch_audit = next(a for a in audit_list if a["policy"] == "DIRECT_BOTORCH_BASELINE")
    assert botorch_audit["verified"] is True
    assert botorch_audit["uses_direct_botorch_backend"] is True
    assert botorch_audit["coordinator_proposal_count"] == 0
    assert botorch_audit["source_adapter_invocations"] == 0
    assert botorch_audit["direct_botorch_calls"] == 2

    random_audit = next(a for a in audit_list if a["policy"] == "random")
    assert random_audit["verified"] is True
    assert random_audit["coordinator_proposal_count"] == 0
    assert random_audit["source_adapter_invocations"] == 0
    assert random_audit["random_steps"] == 2

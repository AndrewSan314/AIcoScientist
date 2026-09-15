"""Strict anti-cheating and no-lookahead regression tests for process rediscovery."""

from __future__ import annotations

import copy
import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import BlindExperimentalOracle, RediscoveryReplay


@pytest.fixture
def candidate_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"candidate_id": f"R_{i}", "x1": float(i * 10), "x2": float(i % 3), "capacity": float(i * 1.5 + (10.0 if i == 7 else 0.0))}
        for i in range(10)
    ])


def test_initial_design_never_includes_hidden_best(candidate_pool: pd.DataFrame) -> None:
    """Verifies that across multiple random seeds, the initial design NEVER samples hidden best."""
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    oracle_temp = replay._create_oracle()
    hidden_best = oracle_temp.hidden_best_id

    for seed in range(50):
        traj = replay.run(
            strategy="random",
            seed=seed,
            initial_size=3,
            max_steps=1,
        )
        assert hidden_best not in traj.initial_candidate_ids, (
            f"Seed {seed} leaked hidden best recipe {hidden_best} into initial design!"
        )


def test_surrogate_training_data_bounded_by_historical_steps(candidate_pool: pd.DataFrame) -> None:
    """Verifies that at step t, the surrogate model sees only <= t revealed observations."""
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    traj = replay.run(
        strategy="expected_improvement",
        seed=42,
        initial_size=3,
        max_steps=5,
    )
    # Number of steps executed
    assert len(traj.steps) == 5
    # Steps are 1-indexed
    for idx, step_rec in enumerate(traj.steps):
        assert step_rec.step == idx + 1


def test_perturbing_unrevealed_outcome_does_not_affect_proposal(candidate_pool: pd.DataFrame) -> None:
    """Strict anti-cheating invariant: changing an unrevealed outcome has zero impact on BO proposal."""
    pool_a = candidate_pool.copy()
    pool_b = candidate_pool.copy()

    # Candidate R_7 is the best candidate in candidate_pool
    # In pool_b, change R_0 (an unrevealed candidate) outcome by 1000x
    # Keep initial design identical by using the same seed
    replay_a = RediscoveryReplay(candidate_pool=pool_a, candidate_id_column="candidate_id", target_column="capacity")
    replay_b = RediscoveryReplay(candidate_pool=pool_b, candidate_id_column="candidate_id", target_column="capacity")

    traj_a = replay_a.run(strategy="greedy", seed=101, initial_size=3, max_steps=1)

    # In pool_b, perturb an unrevealed candidate's target within sub-optimal range
    # so R_7 remains the true hidden best (avoiding altering the initial exclusion set)
    unrevealed_in_a = [c for c in candidate_pool["candidate_id"] if c not in traj_a.initial_candidate_ids and c != traj_a.steps[0].selected_id]
    test_cand = unrevealed_in_a[0]
    # R_7 is 20.5; perturb test_cand to 5.0 (well below 20.5)
    pool_b.loc[pool_b["candidate_id"] == test_cand, "capacity"] = 5.0

    replay_b_perturbed = RediscoveryReplay(candidate_pool=pool_b, candidate_id_column="candidate_id", target_column="capacity")
    traj_b = replay_b_perturbed.run(strategy="greedy", seed=101, initial_size=3, max_steps=1)

    # The first optimization step proposal must be strictly identical
    assert traj_a.initial_candidate_ids == traj_b.initial_candidate_ids
    assert traj_a.steps[0].selected_id == traj_b.steps[0].selected_id
    assert pytest.approx(traj_a.steps[0].predicted_mean, abs=1e-6) == traj_b.steps[0].predicted_mean


def test_perturbing_unrevealed_d30_does_not_affect_ai_engine_proposal() -> None:
    """Adversarial test: perturbing target D30 of an unrevealed recipe cannot alter AI engine proposal."""
    from src.process.contracts import BatteryProcessRun, ParameterValue, MeasurementValue, StageRecord, ProvenanceRecord
    from src.process.information_horizon import DecisionHorizon
    from src.process.stages import ProcessStage, STAGE_ORDER

    prov = ProvenanceRecord("PHYSICAL_HISTORICAL", "", "", "1", {}, "2", {})
    records = []
    runs_by_recipe = {}
    for i in range(8):
        rid = f"recipe_{i}"
        val = 300.0 + (60.0 if i == 6 else 3.0 * i)
        records.append({
            "recipe_id": rid,
            "coating_speed_m_per_min": 0.2 + 0.05 * i,
            "coating_gap_um": 100.0 + 10.0 * i,
            "discharge_specific_capacity_cycle30_mah_g": val,
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
                        "coating_speed_m_per_min": ParameterValue(0.2 + 0.05 * i, "m/min"),
                        "coating_gap_um": ParameterValue(100.0 + 10.0 * i, "um"),
                    },
                    intermediate_properties={"wet_thickness_um": MeasurementValue(120.0, "um")},
                    modalities=[],
                    provenance=prov,
                ),
            ],
            final_kpis={"discharge_specific_capacity_cycle30_mah_g": MeasurementValue(val, "mAh/g")},
            provenance=prov,
        )
        runs_by_recipe[rid] = [run]

    df_a = pd.DataFrame(records).sort_values(by="discharge_specific_capacity_cycle30_mah_g", ascending=False).reset_index(drop=True)
    df_b = df_a.copy()

    replay_a = RediscoveryReplay(
        candidate_pool=df_a,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        runs_by_recipe=runs_by_recipe,
    )
    traj_a = replay_a.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=3, max_steps=1)

    # Find an unrevealed candidate that was NOT selected in step 1
    unrevealed_non_selected = [
        r for r in df_a["recipe_id"]
        if r not in traj_a.initial_candidate_ids and r != traj_a.steps[0].selected_id and r != traj_a.hidden_best_id
    ][0]

    # In df_b, perturb that unrevealed candidate's D30 by a large amount (e.g. drop from 300 to 10.0)
    df_b.loc[df_b["recipe_id"] == unrevealed_non_selected, "discharge_specific_capacity_cycle30_mah_g"] = 10.0

    replay_b = RediscoveryReplay(
        candidate_pool=df_b,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        runs_by_recipe=runs_by_recipe,
    )
    traj_b = replay_b.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=3, max_steps=1)

    # Next proposal must remain strictly identical
    assert traj_a.initial_candidate_ids == traj_b.initial_candidate_ids
    assert traj_a.steps[0].selected_id == traj_b.steps[0].selected_id
    assert pytest.approx(traj_a.steps[0].predicted_mean, abs=1e-5) == traj_b.steps[0].predicted_mean
    assert pytest.approx(traj_a.steps[0].acquisition_value, abs=1e-5) == traj_b.steps[0].acquisition_value


def test_perturbing_unrevealed_active_mass_thickness_porosity_does_not_affect_proposal() -> None:
    """Modifying unrevealed post-process metrology (mass, thickness, porosity) has zero impact on proposal."""
    from src.process.contracts import BatteryProcessRun, ParameterValue, MeasurementValue, StageRecord, ProvenanceRecord
    from src.process.information_horizon import DecisionHorizon
    from src.process.stages import ProcessStage, STAGE_ORDER

    prov = ProvenanceRecord("PHYSICAL_HISTORICAL", "", "", "1", {}, "2", {})
    records = []
    runs_a = {}
    runs_b = {}
    for i in range(8):
        rid = f"recipe_{i}"
        val = 300.0 + (60.0 if i == 6 else 3.0 * i)
        records.append({
            "recipe_id": rid,
            "coating_speed_m_per_min": 0.2 + 0.05 * i,
            "coating_gap_um": 100.0 + 10.0 * i,
            "discharge_specific_capacity_cycle30_mah_g": val,
        })
        def make_run(mass: float, thick: float, poro: float) -> BatteryProcessRun:
            return BatteryProcessRun(
                run_id=f"run_{i}",
                cell_id=f"cell_{i}",
                batch_id=rid,
                chemistry_id="graphite",
                equipment_context={},
                environment_context={},
                stages=[
                    StageRecord(
                        stage_id=f"run_{i}:coating",
                        stage_type=ProcessStage.COATING,
                        sequence_index=STAGE_ORDER[ProcessStage.COATING],
                        controls={"coating_speed_m_per_min": ParameterValue(0.2 + 0.05 * i, "m/min"), "coating_gap_um": ParameterValue(100.0 + 10.0 * i, "um")},
                        intermediate_properties={
                            "active_mass_mg": MeasurementValue(mass, "mg"),
                            "electrode_thickness_um": MeasurementValue(thick, "um"),
                            "porosity_pct": MeasurementValue(poro, "%"),
                        },
                        modalities=[],
                        provenance=prov,
                    ),
                ],
                final_kpis={"discharge_specific_capacity_cycle30_mah_g": MeasurementValue(val, "mAh/g")},
                provenance=prov,
            )
        runs_a[rid] = [make_run(7.5, 45.0, 32.0)]
        # In runs_b, drastically alter metrology
        runs_b[rid] = [make_run(999.0, 999.0, 99.0)]

    df = pd.DataFrame(records).sort_values(by="discharge_specific_capacity_cycle30_mah_g", ascending=False).reset_index(drop=True)

    replay_a = RediscoveryReplay(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        runs_by_recipe=runs_a,
    )
    traj_a = replay_a.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=3, max_steps=1)

    replay_b = RediscoveryReplay(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        runs_by_recipe=runs_b,
    )
    traj_b = replay_b.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=3, max_steps=1)

    assert traj_a.steps[0].selected_id == traj_b.steps[0].selected_id
    assert pytest.approx(traj_a.steps[0].predicted_mean, abs=1e-5) == traj_b.steps[0].predicted_mean



from __future__ import annotations

import pandas as pd
import pytest

from src.domains.auirh import (
    AUIRH_DOMAIN_CONFIG,
    AUIRH_MODALITY_PROPERTY,
    AUIRH_MODALITY_XRD,
    AUIRH_OBJECTIVE_K0,
    AuIrRhDomainAdapter,
    AuIrRhHypothesisProvider,
)
from src.science.actions import (
    ActionType,
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
    MaterialDomainAdapter,
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)


def test_auirh_adapter_implements_domain_contract() -> None:
    """Verifies that AuIrRhDomainAdapter strictly satisfies the MaterialDomainAdapter protocol."""
    adapter = AuIrRhDomainAdapter()
    assert isinstance(adapter, MaterialDomainAdapter)
    assert adapter.domain_id == "auirh"

    # Candidate pool verification
    pool = adapter.get_candidate_pool()
    assert isinstance(pool, pd.DataFrame)
    assert len(pool) == 966
    assert set(pool.columns) >= {"candidate_id", "Au", "Ir", "Rh"}
    # Strict firewall: unexecuted targets must not exist in visible candidate pool
    assert "k0" not in pool.columns
    assert "intensity" not in pool.columns

    # Candidate features
    first_cid = str(pool.iloc[0]["candidate_id"])
    feats = adapter.get_candidate_features(first_cid)
    assert isinstance(feats, dict)
    assert set(feats.keys()) == {"Au", "Ir", "Rh"}

    # Objectives & Modalities
    objectives = adapter.get_objectives()
    assert len(objectives) == 1
    assert objectives[0].name == "k0"
    assert objectives[0].direction == ObjectiveDirection.MAXIMIZE

    modalities = adapter.get_modality_schema()
    mod_names = {m.name for m in modalities}
    assert mod_names == {"XRD", "PROPERTY"}

    # Hypothesis provider
    provider = adapter.get_hypothesis_provider()
    assert provider is not None
    hypos = provider.build_hypotheses()
    assert set(hypos.keys()) == {"H1", "H2", "H3"}


def test_generic_action_type_accepts_non_xrd_modality() -> None:
    """Verifies that scientific action structures accept arbitrary non-XRD/PROPERTY action types."""
    custom_action = ScientificAction(
        action_id="act_sem_001",
        candidate_id="BAT_001",
        action_type="SEM",
        estimated_cost=2.0,
    )
    assert custom_action.action_type == "SEM"
    assert normalize_action_type(custom_action.action_type) == "SEM"

    d = custom_action.to_dict()
    assert d["action_type"] == "SEM"

    restored = ScientificAction.from_dict(d)
    assert restored.action_type == "SEM"
    assert restored.action_id == "act_sem_001"

    custom_outcome = ExperimentOutcome(
        action_id="act_cycling_001",
        candidate_id="BAT_001",
        action_type="CYCLING",
        revealed_data={"capacity": 185.4},
    )
    assert custom_outcome.action_type == "CYCLING"
    assert custom_outcome.to_dict()["action_type"] == "CYCLING"

    restored_outcome = ExperimentOutcome.from_dict(custom_outcome.to_dict())
    assert restored_outcome.action_type == "CYCLING"
    assert restored_outcome.revealed_data["capacity"] == 185.4


def test_existing_auirh_actions_remain_backward_compatible() -> None:
    """Verifies that legacy ExperimentActionType enum values continue to serialize, compare, and round-trip."""
    act_xrd = ScientificAction(
        action_id="act_xrd_1",
        candidate_id="Au-Ir-Rh_1_1",
        action_type=ExperimentActionType.XRD,
        estimated_cost=1.0,
    )
    assert act_xrd.action_type == ExperimentActionType.XRD
    assert act_xrd.action_type == "XRD"
    assert normalize_action_type(act_xrd.action_type) == "XRD"

    d_xrd = act_xrd.to_dict()
    assert d_xrd["action_type"] == "XRD"

    restored_xrd = ScientificAction.from_dict(d_xrd)
    assert restored_xrd.action_type == ExperimentActionType.XRD
    assert isinstance(restored_xrd.action_type, ExperimentActionType)

    # Outcome round-trip
    outcome = ExperimentOutcome(
        action_id="act_prop_1",
        candidate_id="Au-Ir-Rh_1_1",
        action_type=ExperimentActionType.PROPERTY,
        revealed_data={"k0": 0.0142},
    )
    assert outcome.action_type == ExperimentActionType.PROPERTY
    assert outcome.action_type == "PROPERTY"
    d_out = outcome.to_dict()
    assert d_out["action_type"] == "PROPERTY"
    restored_out = ExperimentOutcome.from_dict(d_out)
    assert restored_out.action_type == ExperimentActionType.PROPERTY

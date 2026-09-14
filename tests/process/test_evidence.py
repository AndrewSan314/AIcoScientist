from __future__ import annotations

import pytest

from src.process.evidence import EvidenceAcquisitionPolicy, EvidenceOption, choose_cost_aware_evidence, choose_evidence, replay_blinded_evidence
from src.process.information_horizon import HorizonView
from src.process.stages import ProcessStage


def _option(modality_id: str, stage: ProcessStage = ProcessStage.DRYING, **kwargs: object) -> EvidenceOption:
    return EvidenceOption(
        modality_id, stage, kwargs.pop("available", True), kwargs.pop("cost", 3.0), kwargs.pop("latency_seconds", 2.0),
        reveals_source_observation=kwargs.pop("source", True), **kwargs,
    )


def test_blinded_evidence_reveals_only_selected_source_observation() -> None:
    options = [_option("sem-1"), _option("edx-1")]
    result = replay_blinded_evidence(
        {"process": 1},
        {"sem-1": {"pixels": [1, 2]}, "edx-1": {"spectrum": [3]}},
        options,
        lambda visible, choices: choices[0],
        decision_stage=ProcessStage.CALENDERING,
    )
    assert result.selected_modality_id == "sem-1"
    assert result.visible_observations["sem-1"] == {"pixels": [1, 2]}
    assert "edx-1" not in result.visible_observations
    assert result.acquired_cost == 3.0


def test_blinded_evidence_rejects_future_stage_before_choose() -> None:
    with pytest.raises(ValueError, match="beyond"):
        replay_blinded_evidence({}, {"future": 1}, [_option("future", ProcessStage.CALENDERING)], lambda *_: pytest.fail("chooser must not run"), decision_stage=ProcessStage.DRYING)


@pytest.mark.parametrize("option", [
    _option("final", ProcessStage.FINAL_CHARACTERIZATION),
    _option("unavailable", available=False),
    _option("synthetic", source=False),
])
def test_blinded_evidence_rejects_final_unavailable_and_synthetic_options(option: EvidenceOption) -> None:
    with pytest.raises(ValueError):
        replay_blinded_evidence({}, {option.modality_id: 1}, [option], lambda *_: pytest.fail("chooser must not run"), decision_stage=ProcessStage.CALENDERING)


def test_blinded_evidence_allows_none_without_reveal() -> None:
    result = replay_blinded_evidence({"process": 1}, {}, [_option("sem")], lambda *_: None, decision_stage=ProcessStage.CALENDERING)
    assert result.selected_modality_id is None
    assert result.visible_observations == {"process": 1}
    assert result.acquired_cost == 0.0


def test_blinded_evidence_accepts_a_typed_horizon() -> None:
    horizon = HorizonView(ProcessStage.CALENDERING, {}, {}, (), ())
    result = replay_blinded_evidence({}, {"sem": 1}, [_option("sem")], lambda *_: None, decision_stage=horizon)
    assert result.selected_modality_id is None


def test_blinded_evidence_rejects_none_stage() -> None:
    with pytest.raises(TypeError, match="decision_stage"):
        replay_blinded_evidence({}, {}, [], lambda *_: None, decision_stage=None)  # type: ignore[arg-type]


def test_cost_aware_policy_uses_declared_pre_reveal_utility() -> None:
    cheap = _option("ultrasound", cost=1.0)
    expensive = _option("sem", cost=4.0)
    assert choose_cost_aware_evidence((cheap, expensive), {"ultrasound": 0.5, "sem": 1.0}) == cheap
    with pytest.raises(ValueError, match="exactly"):
        choose_cost_aware_evidence((cheap,), {"withheld": 1.0})


def test_evidence_policy_choices_are_pre_reveal_and_fail_closed() -> None:
    cheap = _option("ultrasound", cost=1.0)
    expensive = _option("sem", cost=4.0)
    assert choose_evidence((cheap, expensive), EvidenceAcquisitionPolicy.MAX_PREDICTIVE_VARIANCE_REDUCTION, estimated_variance_reduction={"ultrasound": 0.5, "sem": 1.0}) == expensive
    assert choose_evidence((cheap, expensive), EvidenceAcquisitionPolicy.EXPECTED_INFORMATION_VALUE_PER_COST, estimated_decision_utility={"ultrasound": 0.5, "sem": 1.0}) == cheap
    assert choose_evidence((cheap,), EvidenceAcquisitionPolicy.NO_ADDITIONAL_EVIDENCE) is None
    with pytest.raises(ValueError, match="unavailable"):
        choose_evidence((_option("hidden", available=False),), EvidenceAcquisitionPolicy.MAX_PREDICTIVE_VARIANCE_REDUCTION, estimated_variance_reduction={"hidden": 1.0})

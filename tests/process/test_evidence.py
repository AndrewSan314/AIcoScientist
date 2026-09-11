from __future__ import annotations

import pytest

from src.process.evidence import EvidenceOption, replay_blinded_evidence
from src.process.information_horizon import HorizonView
from src.process.stages import ProcessStage


def _option(modality_id: str, stage: ProcessStage = ProcessStage.DRYING, **kwargs: object) -> EvidenceOption:
    return EvidenceOption(modality_id, stage, kwargs.pop("available", True), 3.0, 2.0, reveals_source_observation=kwargs.pop("source", True), **kwargs)


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

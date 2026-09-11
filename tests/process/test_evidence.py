from __future__ import annotations

from src.process.evidence import EvidenceOption, replay_blinded_evidence
from src.process.stages import ProcessStage


def test_blinded_evidence_reveals_only_selected_source_observation() -> None:
    option = EvidenceOption("sem-1", ProcessStage.CALENDERING, True, 3.0, 2.0, reveals_source_observation=True)
    result = replay_blinded_evidence(
        {"process": 1},
        {"sem-1": {"pixels": [1, 2]}, "edx-1": {"spectrum": [3]}},
        [option],
        lambda visible, options: options[0],
    )
    assert result.selected_modality_id == "sem-1"
    assert result.visible_observations["sem-1"] == {"pixels": [1, 2]}
    assert "edx-1" not in result.visible_observations
    assert result.acquired_cost == 3.0

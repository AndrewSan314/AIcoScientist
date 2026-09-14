from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .core import SurrogateArtifact, SurrogateDecisionContext


@dataclass(frozen=True)
class OneStageLookaheadRollout:
    mode: str
    stage: str
    context_fingerprint: str
    artifact_fingerprint: str
    candidate_controls: tuple[Mapping[str, float], ...]
    predictions: Mapping[str, Mapping[str, tuple[float, ...]]]
    uncertainty_kind: str


def one_stage_lookahead(
    artifact: SurrogateArtifact,
    context: SurrogateDecisionContext,
    candidate_controls: Sequence[Mapping[str, float]],
    *,
    targets: Sequence[str] | None = None,
    monte_carlo_samples: int | None = None,
) -> OneStageLookaheadRollout:
    """Bounded rollout for stage-specific GP/tree artifacts; never labels itself full horizon."""
    if monte_carlo_samples is not None:
        raise ValueError("stage-specific artifacts support ONE_STAGE_LOOKAHEAD only; no Monte Carlo downstream rollout")
    if not candidate_controls:
        raise ValueError("at least one candidate control setting is required")
    selected = tuple(targets or artifact.target_names)
    if not selected or set(selected) - set(artifact.target_names):
        raise ValueError("rollout targets must be present in the frozen artifact")
    distributions = artifact.predict_decision(context, candidate_controls)
    predictions = {
        target: {
            "mean": tuple(map(float, distributions[target][0])),
            "std": tuple(map(float, np.maximum(distributions[target][1], 0.0))),
        }
        for target in selected
    }
    return OneStageLookaheadRollout(
        "ONE_STAGE_LOOKAHEAD", context.stage.value, context.context_fingerprint, artifact.artifact_fingerprint,
        tuple(dict(controls) for controls in candidate_controls), predictions, artifact.surrogate.uncertainty_kind,
    )

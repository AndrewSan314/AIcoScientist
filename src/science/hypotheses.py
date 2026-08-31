from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class HypothesisStatus(str, Enum):
    """Auditable state of a scientific hypothesis."""

    ACTIVE = "ACTIVE"
    SUPPORTED = "SUPPORTED"
    WEAKENED = "WEAKENED"
    DISCRIMINATED = "DISCRIMINATED"


@dataclass
class ScientificHypothesis:
    """A structured, methodologically testable scientific predictive hypothesis.

    CLAIM BOUNDARY CONTRACT:
    - Represents heuristic, evidence-weighted predictive models on observable data.
    - Does NOT claim causal physical mechanisms, active-site causality, or exact Bayesian posteriors.
    """

    hypothesis_id: str
    title: str
    statement: str
    assumptions: list[str]
    predictive_scope: str
    required_modalities: list[str]
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    belief_score: float = 0.3333333333333333
    raw_evidence_score: float = 0.0
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    falsification_rule: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "statement": self.statement,
            "assumptions": list(self.assumptions),
            "predictive_scope": self.predictive_scope,
            "required_modalities": list(self.required_modalities),
            "status": self.status.value,
            "belief_score": self.belief_score,
            "raw_evidence_score": self.raw_evidence_score,
            "supporting_evidence_count": self.supporting_evidence_count,
            "contradicting_evidence_count": self.contradicting_evidence_count,
            "falsification_rule": self.falsification_rule,
            "metadata": dict(self.metadata),
        }


def get_default_hypotheses() -> dict[str, ScientificHypothesis]:
    """Returns the 3 canonical methodologically testable predictive hypotheses for Au-Ir-Rh materials discovery."""
    h1 = ScientificHypothesis(
        hypothesis_id="H1",
        title="Direct Composition Hypothesis",
        statement="Composition-only explanation is sufficient for predicting observed electrocatalytic rate constant k0 across the ternary composition space.",
        assumptions=[
            "Electrochemical kinetics vary smoothly with nominal elemental composition (Au, Ir, Rh).",
            "Predictive accuracy is not substantially improved by measuring crystallographic diffraction.",
        ],
        predictive_scope="Continuous ternary composition space (Au-Ir-Rh).",
        required_modalities=["composition", "k0"],
        falsification_rule="Evidence against H1 increases when structure-informed models demonstrate out-of-sample predictive advantage over composition-only models on held-out samples.",
    )

    h2 = ScientificHypothesis(
        hypothesis_id="H2",
        title="Structure-Mediated Hypothesis",
        statement="Revealed XRD crystal structure provides predictive information for k0 beyond nominal composition alone.",
        assumptions=[
            "Crystallographic diffraction features capture structural variations across library regions.",
            "Incorporating structural embeddings into property surrogate models reduces generalization error on held-out samples.",
        ],
        predictive_scope="Multimodal structural-compositional regimes.",
        required_modalities=["composition", "xrd", "k0"],
        falsification_rule="Evidence against H2 increases if structure-informed property models yield higher or indistinguishable cross-validation error compared to composition-only models.",
    )

    h3 = ScientificHypothesis(
        hypothesis_id="H3",
        title="Local Structural-Regime Hypothesis",
        statement="Some local composition regions exhibit structural characteristics that are poorly captured by smooth composition-based interpolation.",
        assumptions=[
            "Certain composition neighborhoods experience crystallographic peak shifts or distinct diffraction signatures.",
            "Regions with high structural surrogate prediction uncertainty correlate with elevated structural novelty.",
        ],
        predictive_scope="High-gradient transition boundaries across ternary libraries.",
        required_modalities=["composition", "xrd"],
        falsification_rule="Evidence against H3 increases if XRD measurements in high-uncertainty regions reveal predictable, low-residual diffraction embeddings that match standard interpolation.",
    )

    return {"H1": h1, "H2": h2, "H3": h3}


@dataclass(frozen=True)
class EvidenceEvent:
    """An explicit, immutable scientific observation event."""

    event_id: str
    action_type: str
    candidate_id: str
    structure_residual: float | None = None
    structure_novelty: float | None = None
    property_residual: float | None = None
    structure_advantage_ratio: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HypothesisEngine:
    """Manages event-driven evidence accumulation and evidence-weighted belief updates.

    SCIENTIFIC EVIDENCE INVARIANTS:
    - Belief values represent evidence-weighted softmax normalized scores, NOT exact Bayesian posteriors.
    - Evidence is updated strictly from real sequential events derived from revealed observations.
    - Calling update or refit repeatedly WITHOUT new observations does NOT increment evidence counters.
    """

    def __init__(self, hypotheses: dict[str, ScientificHypothesis] | None = None) -> None:
        self.hypotheses = hypotheses or get_default_hypotheses()
        self.evidence_events: list[EvidenceEvent] = []
        self.update_history: list[dict[str, Any]] = []

    def record_evidence_event(
        self,
        event_id: str,
        action_type: str,
        candidate_id: str,
        structure_residual: float | None = None,
        structure_novelty: float | None = None,
        property_residual: float | None = None,
        structure_advantage_ratio: float = 0.0,
    ) -> dict[str, float]:
        """Records a new empirical observation event and updates hypothesis beliefs.

        This is the ONLY method that increments supporting/contradicting evidence counts.
        """
        event = EvidenceEvent(
            event_id=event_id,
            action_type=action_type,
            candidate_id=candidate_id,
            structure_residual=structure_residual,
            structure_novelty=structure_novelty,
            property_residual=property_residual,
            structure_advantage_ratio=structure_advantage_ratio,
        )
        self.evidence_events.append(event)

        # 1. Update evidence counters based on genuine event observations
        if action_type == "XRD":
            if structure_residual is not None:
                if structure_residual > 0.35:
                    self.hypotheses["H3"].supporting_evidence_count += 1
                else:
                    self.hypotheses["H3"].contradicting_evidence_count += 1

        elif action_type == "PROPERTY":
            if structure_advantage_ratio > 0.05:
                self.hypotheses["H2"].supporting_evidence_count += 1
                self.hypotheses["H1"].contradicting_evidence_count += 1
            elif structure_advantage_ratio < -0.05:
                self.hypotheses["H1"].supporting_evidence_count += 1
                self.hypotheses["H2"].contradicting_evidence_count += 1

        beliefs = self._recompute_beliefs(structure_advantage_ratio=structure_advantage_ratio)

        update_record = {
            "step": len(self.update_history) + 1,
            "event_id": event_id,
            "action_type": action_type,
            "candidate_id": candidate_id,
            "beliefs": {hid: h.belief_score for hid, h in self.hypotheses.items()},
            "raw_scores": {hid: h.raw_evidence_score for hid, h in self.hypotheses.items()},
            "structure_advantage_ratio": structure_advantage_ratio,
        }
        self.update_history.append(update_record)
        return beliefs

    def recalculate_current_scores(self, structure_advantage_ratio: float = 0.0) -> dict[str, float]:
        """Recalculates belief scores from accumulated evidence without appending to update history."""
        return self._recompute_beliefs(structure_advantage_ratio=structure_advantage_ratio)

    def _recompute_beliefs(self, structure_advantage_ratio: float = 0.0) -> dict[str, float]:
        """Recomputes raw evidence scores and softmax normalized beliefs from accumulated events."""
        if not self.evidence_events:
            for h in self.hypotheses.values():
                h.belief_score = 1.0 / len(self.hypotheses)
                h.raw_evidence_score = 0.0
                h.status = HypothesisStatus.ACTIVE
            return {hid: h.belief_score for hid, h in self.hypotheses.items()}

        # Compute empirical averages over recorded events
        struct_res_list = [e.structure_residual for e in self.evidence_events if e.structure_residual is not None]
        struct_nov_list = [e.structure_novelty for e in self.evidence_events if e.structure_novelty is not None]
        prop_res_list = [e.property_residual for e in self.evidence_events if e.property_residual is not None]

        mean_struct_res = float(np.mean(struct_res_list)) if struct_res_list else 0.0
        mean_struct_nov = float(np.mean(struct_nov_list)) if struct_nov_list else 0.0
        mean_prop_res = float(np.mean(prop_res_list)) if prop_res_list else 0.0
        num_xrd = len(struct_res_list)

        # 1. Evidence for H1 (Direct Composition)
        # Favored when property residual is low and structure advantage is low/negative
        e_h1 = max(-5.0, min(5.0, (1.0 - 2.0 * structure_advantage_ratio) - 0.5 * mean_prop_res))

        # 2. Evidence for H2 (Structure-Mediated)
        # Favored when structural advantage is positive from out-of-sample CV
        e_h2 = max(-5.0, min(5.0, (3.0 * structure_advantage_ratio) + (0.5 if num_xrd >= 3 else 0.0)))

        # 3. Evidence for H3 (Local Structural-Regime)
        # Favored when structural novelty and residuals in revealed XRDs are high
        e_h3 = max(-5.0, min(5.0, (2.0 * mean_struct_nov) + (1.5 * mean_struct_res) - 0.5))

        self.hypotheses["H1"].raw_evidence_score = float(e_h1)
        self.hypotheses["H2"].raw_evidence_score = float(e_h2)
        self.hypotheses["H3"].raw_evidence_score = float(e_h3)

        # Softmax normalization
        raw_scores = np.array([e_h1, e_h2, e_h3], dtype=np.float64)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        beliefs = exp_scores / np.sum(exp_scores)

        for i, hid in enumerate(["H1", "H2", "H3"]):
            h = self.hypotheses[hid]
            h.belief_score = float(beliefs[i])
            if h.belief_score > 0.50:
                h.status = HypothesisStatus.SUPPORTED
            elif h.belief_score < 0.20:
                h.status = HypothesisStatus.WEAKENED
            else:
                h.status = HypothesisStatus.ACTIVE

        return {hid: h.belief_score for hid, h in self.hypotheses.items()}

    def get_falsification_criterion(self, hypothesis_id: str) -> str:
        """Returns the specific testable falsification criterion for a given hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return "Observation must quantitatively contradict model predictive distribution."
        return self.hypotheses[hypothesis_id].falsification_rule

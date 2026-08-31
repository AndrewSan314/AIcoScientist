from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    """A structured, methodologically testable scientific hypothesis."""

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
    """Returns the 3 canonical methodologically testable hypotheses for Au-Ir-Rh materials discovery."""
    h1 = ScientificHypothesis(
        hypothesis_id="H1",
        title="Direct Composition Hypothesis",
        statement="Observed electrocatalytic rate constant k0 can be predicted and optimized primarily from elemental composition (Au, Ir, Rh) without requiring structural characterization.",
        assumptions=[
            "Smooth kinetic variation across the ternary composition simplex.",
            "Solid solution behavior dominates without abrupt crystallographic phase boundaries.",
        ],
        predictive_scope="Continuous ternary composition space (Au-Ir-Rh).",
        required_modalities=["composition", "k0"],
        falsification_rule="Evidence against H1 increases when structure-informed surrogate significantly outperforms composition-only surrogate on held-out samples, or when compositionally similar pairs exhibit divergent kinetics explained by distinct XRD diffraction phases.",
    )

    h2 = ScientificHypothesis(
        hypothesis_id="H2",
        title="Structure-Mediated Hypothesis",
        statement="Crystallographic and phase features observable in XRD diffractograms provide predictive information about k0 beyond nominal elemental composition alone.",
        assumptions=[
            "Non-equilibrium synthesis produces phase segregation or order-disorder regimes.",
            "XRD peak positions and intensity distributions directly correlate with catalytic active sites.",
        ],
        predictive_scope="Multimodal structural-compositional regimes.",
        required_modalities=["composition", "xrd", "k0"],
        falsification_rule="Evidence against H2 increases if measured XRD diffractograms show no structural distinction or if structure-informed property predictions remain statistically indistinguishable from composition-only baseline.",
    )

    h3 = ScientificHypothesis(
        hypothesis_id="H3",
        title="Local Structural-Regime Hypothesis",
        statement="Specific localized composition regions contain sharp structural transitions where nominal composition alone is insufficient to infer property behavior.",
        assumptions=[
            "Ternary boundaries (e.g. Au-rich to Ir-rich or Rh-rich transitions) contain localized structural complexity.",
            "High structural surrogate uncertainty indicates uncharted crystallographic phase mixtures.",
        ],
        predictive_scope="High-gradient transition boundaries across ternary libraries.",
        required_modalities=["composition", "xrd"],
        falsification_rule="Evidence against H3 increases if XRD characterization in high-uncertainty boundary regions reveals only routine, well-interpolated solid solution diffraction patterns with low structural novelty.",
    )

    return {"H1": h1, "H2": h2, "H3": h3}


class HypothesisEngine:
    """Manages evidence accumulation and evidence-weighted belief updates for scientific hypotheses.

    SCIENTIFIC EVIDENCE CONTRACT:
    - Belief values represent evidence-weighted softmax normalized scores, NOT exact Bayesian posteriors.
    - Evidence is updated exclusively from real revealed experimental observations in the active campaign.
    """

    def __init__(self, hypotheses: dict[str, ScientificHypothesis] | None = None) -> None:
        self.hypotheses = hypotheses or get_default_hypotheses()
        self.update_history: list[dict[str, Any]] = []

    def update_evidence(
        self,
        num_xrd: int,
        num_prop: int,
        structure_advantage_ratio: float,
        structure_novelty_mean: float,
        structure_residual_norm: float,
        property_residual_norm: float,
    ) -> dict[str, float]:
        """Calculates evidence scores and updates normalized hypothesis beliefs.

        Returns:
            Dict mapping hypothesis_id to new normalized belief score.
        """
        # Baseline uninformative prior if no evidence
        if num_xrd == 0 and num_prop == 0:
            for h in self.hypotheses.values():
                h.belief_score = 1.0 / len(self.hypotheses)
                h.raw_evidence_score = 0.0
            return {hid: h.belief_score for hid, h in self.hypotheses.items()}

        # 1. Evidence for H1 (Direct Composition)
        # Favored when property residual is low and structure advantage is low/negative
        e_h1 = max(-5.0, min(5.0, (1.0 - 2.0 * structure_advantage_ratio) - 0.5 * property_residual_norm))

        # 2. Evidence for H2 (Structure-Mediated)
        # Favored when structural advantage is positive and correlates with revealed XRD observations
        e_h2 = max(-5.0, min(5.0, (3.0 * structure_advantage_ratio) + (0.5 if num_xrd >= 3 else 0.0)))

        # 3. Evidence for H3 (Local Structural-Regime)
        # Favored when structural novelty and residuals in revealed XRDs are high
        e_h3 = max(-5.0, min(5.0, (2.0 * structure_novelty_mean) + (1.5 * structure_residual_norm) - 0.5))

        self.hypotheses["H1"].raw_evidence_score = float(e_h1)
        self.hypotheses["H2"].raw_evidence_score = float(e_h2)
        self.hypotheses["H3"].raw_evidence_score = float(e_h3)

        # Update evidence counters
        if structure_advantage_ratio > 0.05:
            self.hypotheses["H2"].supporting_evidence_count += 1
            self.hypotheses["H1"].contradicting_evidence_count += 1
        else:
            self.hypotheses["H1"].supporting_evidence_count += 1
            self.hypotheses["H2"].contradicting_evidence_count += 1

        if structure_novelty_mean > 0.3:
            self.hypotheses["H3"].supporting_evidence_count += 1
        else:
            self.hypotheses["H3"].contradicting_evidence_count += 1

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

        update_record = {
            "step": len(self.update_history) + 1,
            "beliefs": {hid: h.belief_score for hid, h in self.hypotheses.items()},
            "raw_scores": {hid: h.raw_evidence_score for hid, h in self.hypotheses.items()},
            "structure_advantage_ratio": structure_advantage_ratio,
        }
        self.update_history.append(update_record)

        return {hid: h.belief_score for hid, h in self.hypotheses.items()}

    def get_falsification_criterion(self, hypothesis_id: str) -> str:
        """Returns the specific testable falsification criterion for a given hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return "Observation must quantitatively contradict model predictive distribution."
        return self.hypotheses[hypothesis_id].falsification_rule

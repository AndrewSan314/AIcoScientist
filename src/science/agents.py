from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.science.actions import ActionRecommendation, ExperimentActionType, normalize_action_type
from src.science.hypotheses import HypothesisEngine, ScientificHypothesis


@dataclass(frozen=True)
class AgentPerspective:
    """Structured rationale emitted by a specialized scientific agent role."""

    role_name: str
    headline: str
    body: str
    key_points: list[str]


class HypothesisScientistAgent:
    """Synthesizes which competing scientific hypothesis currently demands empirical testing."""

    @property
    def role_name(self) -> str:
        return "Hypothesis Scientist"

    def reason(
        self,
        hypothesis_engine: HypothesisEngine,
        recommendation: ActionRecommendation,
    ) -> AgentPerspective:
        beliefs = hypothesis_engine.hypotheses
        target_h = beliefs[recommendation.hypothesis_id]

        headline = f"Targeting {target_h.title} ({recommendation.hypothesis_id}) with {target_h.belief_score*100:.1f}% Evidence Weight"
        body = (
            f"Current scientific belief is distributed across 3 competing models: "
            f"H1 ({beliefs['H1'].belief_score*100:.1f}%), H2 ({beliefs['H2'].belief_score*100:.1f}%), and H3 ({beliefs['H3'].belief_score*100:.1f}%). "
            f"Hypothesis {target_h.hypothesis_id} asserts: '{target_h.statement}' "
            f"Running {recommendation.action.action_type.value} on candidate {recommendation.action.candidate_id} yields the highest estimated scientific value under the current policy to update this belief."
        )

        return AgentPerspective(
            role_name=self.role_name,
            headline=headline,
            body=body,
            key_points=[
                f"H1 Direct Composition: {beliefs['H1'].belief_score*100:.1f}% evidence weight",
                f"H2 Structure-Mediated: {beliefs['H2'].belief_score*100:.1f}% evidence weight",
                f"H3 Local Structural-Regime: {beliefs['H3'].belief_score*100:.1f}% evidence weight",
            ],
        )


class FalsificationScientistAgent:
    """Defines explicit quantitative criteria that would contradict or weaken the target hypothesis."""

    @property
    def role_name(self) -> str:
        return "Falsification Scientist"

    def reason(
        self,
        recommendation: ActionRecommendation,
        hypothesis_engine: HypothesisEngine,
    ) -> AgentPerspective:
        target_h = hypothesis_engine.hypotheses[recommendation.hypothesis_id]
        headline = f"Falsification Condition for {target_h.title}"
        body = (
            f"To guard against confirmation bias, we pre-specify falsification criteria prior to measurement: "
            f"{recommendation.falsification_criterion}"
        )

        if recommendation.action.action_type == ExperimentActionType.XRD:
            points = [
                "Observed XRD embedding falls within the predictive distribution of the composition-to-structure surrogate.",
                "Observed structural residual is low despite high pre-measurement uncertainty.",
            ]
        else:
            points = [
                "Structure-informed cross-validation does not improve predictive error over composition-only modeling on held-out samples.",
                "Observed electrochemical property k0 aligns with nominal composition predictions.",
            ]

        return AgentPerspective(
            role_name=self.role_name,
            headline=headline,
            body=body,
            key_points=points,
        )


class ExperimentDesignerAgent:
    """Selects and justifies the optimal action and candidate over alternative options."""

    @property
    def role_name(self) -> str:
        return "Experiment Designer"

    def reason(
        self,
        recommendation: ActionRecommendation,
    ) -> AgentPerspective:
        action_title = "XRD Characterization" if recommendation.action.action_type == ExperimentActionType.XRD else "SECCM Electrochemical Test"
        headline = f"Selected {action_title} for {recommendation.action.candidate_id}"
        body = (
            f"Recommended {action_title} with highest estimated scientific value score {recommendation.total_value:.3f} "
            f"(Information Score: {recommendation.scientific_information_value:.3f}, Discovery Score: {recommendation.discovery_value:.3f}, Cost Penalty: {recommendation.cost_penalty:.3f})."
        )

        act_str = normalize_action_type(recommendation.action.action_type)
        contrast_points = [
            f"Recommended: {act_str} on {recommendation.action.candidate_id} (Net Value: {recommendation.total_value:.2f})"
        ]
        for alt in recommendation.alternatives:
            alt_act_str = normalize_action_type(alt.action_type)
            contrast_points.append(
                f"Alternative: {alt_act_str} on {alt.candidate_id} -> {alt.contrastive_rationale}"
            )

        return AgentPerspective(
            role_name=self.role_name,
            headline=headline,
            body=body,
            key_points=contrast_points,
        )


class EvidenceProvenanceAgent:
    """Reports auditable data provenance and boundary conditions for scientific recommendations."""

    @property
    def role_name(self) -> str:
        return "Evidence Provenance"

    def reason(
        self,
        num_xrd_revealed: int,
        num_property_revealed: int,
        total_candidates: int,
    ) -> AgentPerspective:
        headline = f"Data Provenance: {num_xrd_revealed} XRD, {num_property_revealed} Property Observations"
        body = (
            f"Decision inputs: nominal composition for {total_candidates} candidates, "
            f"{num_xrd_revealed} revealed XRD diffractograms, and {num_property_revealed} revealed electrochemical property observations. "
            f"All surrogate predictions and hypothesis evidence weights are derived strictly from these revealed data."
        )

        return AgentPerspective(
            role_name=self.role_name,
            headline=headline,
            body=body,
            key_points=[
                f"Visible candidate pool: {total_candidates} physical samples (Au, Ir, Rh)",
                f"Revealed structural observations: {num_xrd_revealed} samples",
                f"Revealed property observations: {num_property_revealed} samples",
                "Claim boundary: Evidence-weighted heuristic policy; no unverified physical mechanisms or exact Bayesian posteriors claimed.",
            ],
        )


# Backward-compatible alias
EvidenceAuditorAgent = EvidenceProvenanceAgent


class MultiAgentPresentationLayer:
    """Coordinates multi-agent reasoning to generate transparent, auditable perspectives."""

    def __init__(self) -> None:
        self.hypothesis_scientist = HypothesisScientistAgent()
        self.falsification_scientist = FalsificationScientistAgent()
        self.experiment_designer = ExperimentDesignerAgent()
        self.evidence_provenance = EvidenceProvenanceAgent()

    def generate_perspectives(
        self,
        recommendation: ActionRecommendation,
        hypothesis_engine: HypothesisEngine,
        num_xrd_revealed: int,
        num_property_revealed: int,
        total_candidates: int,
    ) -> list[AgentPerspective]:
        """Generates structured perspectives from all specialized agent roles."""
        return [
            self.hypothesis_scientist.reason(hypothesis_engine, recommendation),
            self.falsification_scientist.reason(recommendation, hypothesis_engine),
            self.experiment_designer.reason(recommendation),
            self.evidence_provenance.reason(num_xrd_revealed, num_property_revealed, total_candidates),
        ]

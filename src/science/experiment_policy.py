from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.science.actions import (
    ActionRecommendation,
    CounterfactualAlternative,
    ExperimentActionType,
    ScientificAction,
)
from src.science.hypotheses import HypothesisEngine
from src.science.scientific_models import PropertySurrogateModel, StructureSurrogateModel
from src.science.xrd_representation import XRDRepresentationExtractor

logger = logging.getLogger(__name__)

DEFAULT_XRD_COST = 1.0
DEFAULT_PROPERTY_COST = 5.0
DEFAULT_WEIGHT_INFO = 1.0
DEFAULT_WEIGHT_DISCOVERY = 1.0
DEFAULT_WEIGHT_COST = 0.15


class NextBestExperimentPolicy:
    """Adaptive Next-Best-Experiment Scientific Decision Policy.

    Evaluates the joint discrete action space:
        A = { XRD(c) for c in unobserved_xrd } U { PROPERTY(c) for c in unobserved_property }

    SCIENTIFIC VALUE FORMULATION:
        TOTAL_VALUE(a) = w_info * INFO_VALUE(a) + w_disc * DISCOVERY_VALUE(a) - w_cost * COST(a)

    Generates auditable counterfactual contrastive rationales for alternative candidate/action pairs.
    """

    def __init__(
        self,
        cost_xrd: float = DEFAULT_XRD_COST,
        cost_property: float = DEFAULT_PROPERTY_COST,
        w_info: float = DEFAULT_WEIGHT_INFO,
        w_disc: float = DEFAULT_WEIGHT_DISCOVERY,
        w_cost: float = DEFAULT_WEIGHT_COST,
    ) -> None:
        self.cost_xrd = cost_xrd
        self.cost_property = cost_property
        self.w_info = w_info
        self.w_disc = w_disc
        self.w_cost = w_cost

    def evaluate_actions(
        self,
        candidate_pool_df: pd.DataFrame,
        observed_xrd_ids: set[str],
        observed_property_ids: set[str],
        structure_model: StructureSurrogateModel,
        property_model: PropertySurrogateModel,
        hypothesis_engine: HypothesisEngine,
        step: int = 0,
    ) -> list[dict[str, Any]]:
        """Evaluates and ranks all valid scientific actions."""
        comp_cols = ["Au", "Ir", "Rh"]
        all_comps = candidate_pool_df[comp_cols].to_numpy(dtype=np.float64)
        cids = candidate_pool_df["candidate_id"].tolist()

        # Compute model predictions across all candidates
        _, struct_stds = structure_model.predict(all_comps)
        prop_means, prop_stds = property_model.predict(all_comps)

        # Scale predictions for normalized scoring
        s_std_norm = (struct_stds - np.min(struct_stds)) / (np.max(struct_stds) - np.min(struct_stds) + 1e-12)
        p_std_norm = (prop_stds - np.min(prop_stds)) / (np.max(prop_stds) - np.min(prop_stds) + 1e-12)
        p_mean_norm = (prop_means - np.min(prop_means)) / (np.max(prop_means) - np.min(prop_means) + 1e-12)

        # Hypothesis belief multipliers
        b_h1 = hypothesis_engine.hypotheses["H1"].belief_score
        b_h2 = hypothesis_engine.hypotheses["H2"].belief_score
        b_h3 = hypothesis_engine.hypotheses["H3"].belief_score

        scored_actions: list[dict[str, Any]] = []

        # 1. Evaluate XRD Actions
        for i, cid in enumerate(cids):
            if cid in observed_xrd_ids:
                continue

            # XRD Information Value: structural uncertainty weighted by H2 & H3 importance
            u_struct = float(s_std_norm[i])
            info_val = u_struct * (1.2 * b_h3 + 1.0 * b_h2 + 0.4)
            disc_val = 0.0  # XRD does not directly measure electrochemical k0
            cost = self.cost_xrd
            total = (self.w_info * info_val) + (self.w_disc * disc_val) - (self.w_cost * cost)

            target_h = "H3" if u_struct > 0.6 else "H2"
            scored_actions.append(
                {
                    "candidate_id": cid,
                    "action_type": ExperimentActionType.XRD,
                    "total_value": float(total),
                    "scientific_information_value": float(info_val),
                    "discovery_value": float(disc_val),
                    "cost_penalty": float(self.w_cost * cost),
                    "raw_cost": float(cost),
                    "hypothesis_id": target_h,
                    "struct_uncertainty": u_struct,
                    "prop_uncertainty": float(p_std_norm[i]),
                    "predicted_k0_mean": float(prop_means[i]),
                    "predicted_k0_norm": float(p_mean_norm[i]),
                }
            )

        # 2. Evaluate PROPERTY Actions
        for i, cid in enumerate(cids):
            if cid in observed_property_ids:
                continue

            # Property Discovery Value: Upper Confidence Bound / predicted mean
            u_prop = float(p_std_norm[i])
            pred_val = float(p_mean_norm[i])
            disc_val = pred_val + 0.5 * u_prop

            # Property Information Value: property uncertainty + hypothesis verification
            has_xrd = cid in observed_xrd_ids
            # If XRD was previously revealed, property test validates structure-property coupling
            info_val = u_prop * (1.0 * b_h1 + (1.3 * b_h2 if has_xrd else 0.8 * b_h2) + 0.3)
            cost = self.cost_property
            total = (self.w_info * info_val) + (self.w_disc * disc_val) - (self.w_cost * cost)

            target_h = "H1" if b_h1 > b_h2 else ("H2" if has_xrd else "H1")
            scored_actions.append(
                {
                    "candidate_id": cid,
                    "action_type": ExperimentActionType.PROPERTY,
                    "total_value": float(total),
                    "scientific_information_value": float(info_val),
                    "discovery_value": float(disc_val),
                    "cost_penalty": float(self.w_cost * cost),
                    "raw_cost": float(cost),
                    "hypothesis_id": target_h,
                    "struct_uncertainty": float(s_std_norm[i]),
                    "prop_uncertainty": u_prop,
                    "predicted_k0_mean": float(prop_means[i]),
                    "predicted_k0_norm": pred_val,
                }
            )

        # Sort descending by total value
        scored_actions.sort(key=lambda x: x["total_value"], reverse=True)
        return scored_actions

    def recommend_next_experiment(
        self,
        candidate_pool_df: pd.DataFrame,
        observed_xrd_ids: set[str],
        observed_property_ids: set[str],
        structure_model: StructureSurrogateModel,
        property_model: PropertySurrogateModel,
        hypothesis_engine: HypothesisEngine,
        step: int = 0,
    ) -> ActionRecommendation:
        """Selects the top-ranked next experiment and generates contrastive counterfactuals."""
        scored = self.evaluate_actions(
            candidate_pool_df=candidate_pool_df,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_property_ids,
            structure_model=structure_model,
            property_model=property_model,
            hypothesis_engine=hypothesis_engine,
            step=step,
        )

        if not scored:
            raise RuntimeError("All candidate experiments in the physical library have been exhausted.")

        top = scored[0]
        act = ScientificAction(
            action_id=f"rec_step_{step+1:03d}_{top['action_type'].value}_{top['candidate_id']}",
            candidate_id=top["candidate_id"],
            action_type=top["action_type"],
            estimated_cost=top["raw_cost"],
            requested_at_step=step + 1,
            metadata={
                "total_value": top["total_value"],
                "hypothesis_id": top["hypothesis_id"],
            },
        )

        # Generate Counterfactual Alternatives (at least top 2 alternatives with different traits)
        alternatives: list[CounterfactualAlternative] = []
        for alt in scored[1:]:
            if len(alternatives) >= 2:
                break

            # Build contrastive rationale
            if alt["action_type"] != top["action_type"]:
                contrast = (
                    f"Action {alt['action_type'].value} on {alt['candidate_id']} has "
                    f"{'high predicted discovery value (' + f'{alt['discovery_value']:.2f}' + ')' if alt['action_type'] == ExperimentActionType.PROPERTY else 'lower cost'} "
                    f"but net value ({alt['total_value']:.2f}) is lower than recommended {top['action_type'].value} ({top['total_value']:.2f})."
                )
            else:
                contrast = (
                    f"Candidate {alt['candidate_id']} has similar action type but lower net information score "
                    f"({alt['scientific_information_value']:.2f} vs {top['scientific_information_value']:.2f})."
                )

            alternatives.append(
                CounterfactualAlternative(
                    candidate_id=alt["candidate_id"],
                    action_type=alt["action_type"],
                    total_value=alt["total_value"],
                    scientific_information_value=alt["scientific_information_value"],
                    discovery_value=alt["discovery_value"],
                    cost_penalty=alt["cost_penalty"],
                    hypothesis_id=alt["hypothesis_id"],
                    contrastive_rationale=contrast,
                )
            )

        falsification = hypothesis_engine.get_falsification_criterion(top["hypothesis_id"])
        action_name = "XRD Characterization" if top["action_type"] == ExperimentActionType.XRD else "SECCM Performance Test"
        rationale = (
            f"Selects {action_name} for candidate '{top['candidate_id']}' (Total Scientific Value: {top['total_value']:.3f}). "
            f"Information value ({top['scientific_information_value']:.3f}) and discovery value ({top['discovery_value']:.3f}) "
            f"maximize reduction of uncertainty under test of hypothesis {top['hypothesis_id']}."
        )

        return ActionRecommendation(
            action=act,
            total_value=top["total_value"],
            scientific_information_value=top["scientific_information_value"],
            discovery_value=top["discovery_value"],
            cost_penalty=top["cost_penalty"],
            hypothesis_id=top["hypothesis_id"],
            rationale=rationale,
            falsification_criterion=falsification,
            supporting_evidence=[
                f"Structure uncertainty: {top['struct_uncertainty']:.2f}",
                f"Property uncertainty: {top['prop_uncertainty']:.2f}",
                f"Predicted k0: {top['predicted_k0_mean']:.5f} cm/s",
            ],
            uncertainty_summary={
                "structure_uncertainty": top["struct_uncertainty"],
                "property_uncertainty": top["prop_uncertainty"],
            },
            alternatives=alternatives,
        )

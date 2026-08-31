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
DEFAULT_WEIGHT_DISCOVERY = 0.8
DEFAULT_WEIGHT_COST = 0.8


class NextBestExperimentPolicy:
    """Adaptive Next-Best-Experiment Scientific Decision Policy.

    Evaluates the joint discrete action space:
        A = { XRD(c) for c in unobserved_xrd } U { PROPERTY(c) for c in unobserved_property }

    SCIENTIFIC VALUE FORMULATION:
        TOTAL_VALUE(a) = w_info * INFO_NORM(a) + w_disc * DISC_NORM(a) - w_cost * COST_NORM(a)

    Where:
    - INFO_NORM(a) is the min-max normalized information score across all candidate actions in A.
    - DISC_NORM(a) is the min-max normalized discovery score (0 for XRD; UCB/acquisition for PROPERTY).
    - COST_NORM(a) = raw_cost(a) / max_cost.
    """

    def __init__(
        self,
        cost_xrd: float = DEFAULT_XRD_COST,
        cost_property: float = DEFAULT_PROPERTY_COST,
        w_info: float = DEFAULT_WEIGHT_INFO,
        w_disc: float = DEFAULT_WEIGHT_DISCOVERY,
        w_cost: float = DEFAULT_WEIGHT_COST,
    ) -> None:
        self.cost_xrd = float(cost_xrd)
        self.cost_property = float(cost_property)
        self.w_info = float(w_info)
        self.w_disc = float(w_disc)
        self.w_cost = float(w_cost)

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

        candidate_actions: list[dict[str, Any]] = []

        # 1. Collect candidate XRD actions
        for i, cid in enumerate(cids):
            if cid in observed_xrd_ids:
                continue

            u_struct = float(s_std_norm[i])
            info_val = u_struct * (1.2 * b_h3 + 1.0 * b_h2 + 0.4)
            disc_val = 0.0  # XRD characterization does not measure property k0
            cost = self.cost_xrd
            target_h = "H3" if u_struct > 0.6 else "H2"

            candidate_actions.append(
                {
                    "candidate_id": cid,
                    "action_type": ExperimentActionType.XRD,
                    "raw_info": float(info_val),
                    "raw_disc": float(disc_val),
                    "raw_cost": float(cost),
                    "hypothesis_id": target_h,
                    "struct_uncertainty": u_struct,
                    "prop_uncertainty": float(p_std_norm[i]),
                    "predicted_k0_mean": float(prop_means[i]),
                    "predicted_k0_norm": float(p_mean_norm[i]),
                }
            )

        # 2. Collect candidate PROPERTY actions
        for i, cid in enumerate(cids):
            if cid in observed_property_ids:
                continue

            u_prop = float(p_std_norm[i])
            pred_val = float(p_mean_norm[i])
            disc_val = pred_val + 0.5 * u_prop

            has_xrd = cid in observed_xrd_ids
            info_val = u_prop * (1.0 * b_h1 + (1.3 * b_h2 if has_xrd else 0.8 * b_h2) + 0.3)
            cost = self.cost_property
            target_h = "H1" if b_h1 > b_h2 else ("H2" if has_xrd else "H1")

            candidate_actions.append(
                {
                    "candidate_id": cid,
                    "action_type": ExperimentActionType.PROPERTY,
                    "raw_info": float(info_val),
                    "raw_disc": float(disc_val),
                    "raw_cost": float(cost),
                    "hypothesis_id": target_h,
                    "struct_uncertainty": float(s_std_norm[i]),
                    "prop_uncertainty": u_prop,
                    "predicted_k0_mean": float(prop_means[i]),
                    "predicted_k0_norm": pred_val,
                }
            )

        if not candidate_actions:
            return []

        # 3. Normalized multi-component scoring
        all_info = [a["raw_info"] for a in candidate_actions]
        all_disc = [a["raw_disc"] for a in candidate_actions]
        min_info, max_info = min(all_info), max(all_info)
        min_disc, max_disc = min(all_disc), max(all_disc)
        max_cost = max(self.cost_xrd, self.cost_property, 1.0)

        scored_actions: list[dict[str, Any]] = []
        for a in candidate_actions:
            info_norm = (a["raw_info"] - min_info) / (max_info - min_info + 1e-12)
            disc_norm = (a["raw_disc"] - min_disc) / (max_disc - min_disc + 1e-12)
            cost_norm = a["raw_cost"] / max_cost

            total = (self.w_info * info_norm) + (self.w_disc * disc_norm) - (self.w_cost * cost_norm)

            a["total_value"] = float(total)
            a["scientific_information_value"] = float(info_norm)
            a["discovery_value"] = float(disc_norm)
            a["cost_penalty"] = float(self.w_cost * cost_norm)
            scored_actions.append(a)

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

            # Build contrastive rationale without nested f-strings (Python 3.11 safe)
            if alt["action_type"] != top["action_type"]:
                if alt["action_type"] == ExperimentActionType.PROPERTY:
                    trait = f"high predicted discovery value ({alt['discovery_value']:.2f})"
                else:
                    trait = "lower measurement cost"
                contrast = (
                    f"Action {alt['action_type'].value} on {alt['candidate_id']} has {trait} "
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
            f"Information score ({top['scientific_information_value']:.3f}) and discovery score ({top['discovery_value']:.3f}) "
            f"yield the highest estimated scientific value under the current policy to test hypothesis {top['hypothesis_id']}."
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

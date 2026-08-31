from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.science.actions import (
    ActionRecommendation,
    CounterfactualAlternative,
    ExperimentActionType,
    ScientificAction,
)
from src.science.falsification.information_gain import (
    DiscriminationEvaluation,
    HypothesisInformationGainEstimator,
)
from src.science.hypothesis_models import HypothesisEnsemble

logger = logging.getLogger(__name__)


class FalsificationPolicyMode(str, Enum):
    """Operational mode for the scientific experiment selection policy."""

    PURE_FALSIFICATION = "pure_falsification"
    DISCOVERY_ONLY = "discovery_only"
    HYBRID = "hybrid"


class FalsificationFirstPolicy:
    """Falsification-First Scientific Experiment Selection Policy.

    Selects experiments from the joint discrete action space:
        A = { XRD(c) for c in unobserved_xrd } U { PROPERTY(c) for c in unobserved_property }

    SCIENTIFIC OBJECTIVE MODES:
    1. PURE_FALSIFICATION:
       Score(a) = HIG(a) / (Cost(a) ** gamma)
       Optimizes purely for hypothesis discrimination and falsification efficiency.

    2. DISCOVERY_ONLY:
       Score(a) = BoTorch_Acquisition(a) (0 for XRD) - Cost_norm(a)
       Commodity Bayesian optimization targeting high-performing materials.

    3. HYBRID:
       Score(a) = w_hig * HIG_norm(a) + w_disc * Disc_norm(a) - w_cost * Cost_norm(a)
       Balances epistemic understanding against physical material discovery.
    """

    def __init__(
        self,
        mode: FalsificationPolicyMode | str = FalsificationPolicyMode.HYBRID,
        cost_xrd: float = 1.0,
        cost_property: float = 5.0,
        w_hig: float = 1.0,
        w_disc: float = 0.8,
        w_cost: float = 0.8,
        cost_exponent: float = 1.0,
    ) -> None:
        self.mode = FalsificationPolicyMode(mode)
        self.cost_xrd = float(cost_xrd)
        self.cost_property = float(cost_property)
        self.w_hig = float(w_hig)
        self.w_disc = float(w_disc)
        self.w_cost = float(w_cost)
        self.cost_exponent = float(cost_exponent)
        self.hig_estimator = HypothesisInformationGainEstimator()

    def evaluate_all_actions(
        self,
        candidate_pool_df: pd.DataFrame,
        observed_xrd_ids: set[str],
        observed_property_ids: set[str],
        ensemble: HypothesisEnsemble,
        property_discovery_scores: Mapping[str, float] | None = None,
        observed_xrd_embeddings_map: Mapping[str, np.ndarray] | None = None,
        fast_mode: bool = False,
        seed: int | None = None,
        step: int = 0,
    ) -> list[dict[str, Any]]:
        """Evaluates and ranks all valid candidate actions under current scientific hypotheses."""
        comp_cols = ["Au", "Ir", "Rh"]
        cids = candidate_pool_df["candidate_id"].tolist()
        comps = candidate_pool_df[comp_cols].to_numpy(dtype=np.float64)
        disc_scores = property_discovery_scores or {}
        xrd_embs_map = observed_xrd_embeddings_map or {}

        candidate_actions: list[dict[str, Any]] = []

        # 1. Evaluate valid XRD actions
        for i, cid in enumerate(cids):
            if cid in observed_xrd_ids:
                continue

            eval_res = self.hig_estimator.evaluate_action_discrimination(
                candidate_id=cid,
                action_type=ExperimentActionType.XRD,
                composition=comps[i],
                ensemble=ensemble,
                fast_mode=fast_mode,
                seed=seed,
            )

            candidate_actions.append(
                {
                    "candidate_id": cid,
                    "action_type": ExperimentActionType.XRD,
                    "raw_hig": eval_res.hypothesis_information_gain,
                    "raw_disc": 0.0,  # Characterization does not measure property k0
                    "raw_cost": self.cost_xrd,
                    "property_disagreement": eval_res.property_disagreement,
                    "structure_disagreement": eval_res.structure_disagreement,
                    "current_entropy": eval_res.current_entropy,
                    "expected_entropy": eval_res.expected_posterior_entropy,
                    "predictions": eval_res.predictions,
                }
            )

        # 2. Evaluate valid PROPERTY actions
        for i, cid in enumerate(cids):
            if cid in observed_property_ids:
                continue

            obs_emb = xrd_embs_map.get(cid)
            eval_res = self.hig_estimator.evaluate_action_discrimination(
                candidate_id=cid,
                action_type=ExperimentActionType.PROPERTY,
                composition=comps[i],
                ensemble=ensemble,
                observed_xrd_embedding=obs_emb,
                fast_mode=fast_mode,
                seed=seed,
            )

            raw_botorch_acq = float(disc_scores.get(cid, 0.0))

            candidate_actions.append(
                {
                    "candidate_id": cid,
                    "action_type": ExperimentActionType.PROPERTY,
                    "raw_hig": eval_res.hypothesis_information_gain,
                    "raw_disc": raw_botorch_acq,
                    "raw_cost": self.cost_property,
                    "property_disagreement": eval_res.property_disagreement,
                    "structure_disagreement": eval_res.structure_disagreement,
                    "current_entropy": eval_res.current_entropy,
                    "expected_entropy": eval_res.expected_posterior_entropy,
                    "predictions": eval_res.predictions,
                }
            )

        if not candidate_actions:
            return []

        # 3. Normalization and Mode-Specific Scoring
        all_higs = [a["raw_hig"] for a in candidate_actions]
        min_hig, max_hig = min(all_higs), max(all_higs)

        prop_discs = [a["raw_disc"] for a in candidate_actions if a["action_type"] == ExperimentActionType.PROPERTY]
        min_p_disc = min(prop_discs) if prop_discs else 0.0
        max_p_disc = max(prop_discs) if prop_discs else 1.0

        max_cost = max(self.cost_xrd, self.cost_property, 1.0)

        scored_actions: list[dict[str, Any]] = []
        for a in candidate_actions:
            # Min-max normalized HIG
            hig_norm = (a["raw_hig"] - min_hig) / (max_hig - min_hig + 1e-12) if max_hig > min_hig else a["raw_hig"]

            # Min-max normalized Discovery score (0 for XRD)
            if a["action_type"] == ExperimentActionType.PROPERTY and prop_discs and max_p_disc > min_p_disc:
                disc_norm = (a["raw_disc"] - min_p_disc) / (max_p_disc - min_p_disc + 1e-12)
            else:
                disc_norm = 0.0

            cost_norm = a["raw_cost"] / max_cost

            # Score by policy mode
            if self.mode == FalsificationPolicyMode.PURE_FALSIFICATION:
                total_val = float(a["raw_hig"] / (a["raw_cost"] ** self.cost_exponent))
            elif self.mode == FalsificationPolicyMode.DISCOVERY_ONLY:
                total_val = float(disc_norm - 0.1 * cost_norm)
            else:  # HYBRID
                total_val = float(
                    (self.w_hig * hig_norm) + (self.w_disc * disc_norm) - (self.w_cost * cost_norm)
                )

            a["total_value"] = total_val
            a["normalized_hig"] = float(hig_norm)
            a["normalized_disc"] = float(disc_norm)
            a["normalized_cost"] = float(cost_norm)
            a["cost_penalty"] = float(self.w_cost * cost_norm)
            scored_actions.append(a)

        # Sort descending by total score
        scored_actions.sort(key=lambda x: x["total_value"], reverse=True)
        return scored_actions

    def recommend_next_experiment(
        self,
        candidate_pool_df: pd.DataFrame,
        observed_xrd_ids: set[str],
        observed_property_ids: set[str],
        ensemble: HypothesisEnsemble,
        property_discovery_scores: Mapping[str, float] | None = None,
        observed_xrd_embeddings_map: Mapping[str, np.ndarray] | None = None,
        fast_mode: bool = False,
        seed: int | None = None,
        step: int = 0,
    ) -> ActionRecommendation:
        """Selects top-ranked next experiment, generating contrastive counterfactuals and falsification criteria."""
        scored = self.evaluate_all_actions(
            candidate_pool_df=candidate_pool_df,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_property_ids,
            ensemble=ensemble,
            property_discovery_scores=property_discovery_scores,
            observed_xrd_embeddings_map=observed_xrd_embeddings_map,
            fast_mode=fast_mode,
            seed=seed,
            step=step,
        )

        if not scored:
            raise RuntimeError("All candidate experiments have been exhausted.")

        top = scored[0]
        # Identify which hypothesis is most targeted/uncertain
        beliefs = ensemble.get_beliefs()
        dominant_h = max(beliefs.keys(), key=lambda h: beliefs[h])

        act = ScientificAction(
            action_id=f"rec_step_{step+1:03d}_{top['action_type'].value}_{top['candidate_id']}",
            candidate_id=top["candidate_id"],
            action_type=top["action_type"],
            estimated_cost=top["raw_cost"],
            requested_at_step=step + 1,
            metadata={
                "total_value": top["total_value"],
                "hypothesis_information_gain": top["raw_hig"],
                "policy_mode": self.mode.value,
                "hypothesis_id": dominant_h,
            },
        )

        # Build contrastive counterfactual alternatives
        alternatives: list[CounterfactualAlternative] = []
        for alt in scored[1:]:
            if len(alternatives) >= 2:
                break

            if alt["action_type"] != top["action_type"]:
                if alt["action_type"] == ExperimentActionType.PROPERTY:
                    trait = f"higher predicted discovery score ({alt['normalized_disc']:.2f})"
                else:
                    trait = "lower measurement cost"
                contrast = (
                    f"Action {alt['action_type'].value} on {alt['candidate_id']} has {trait} "
                    f"but net scientific value ({alt['total_value']:.2f}) is lower than recommended {top['action_type'].value} ({top['total_value']:.2f})."
                )
            else:
                contrast = (
                    f"Candidate {alt['candidate_id']} has similar action type but lower Expected Hypothesis Information Gain "
                    f"({alt['raw_hig']:.3f} vs {top['raw_hig']:.3f} nats)."
                )

            alternatives.append(
                CounterfactualAlternative(
                    candidate_id=alt["candidate_id"],
                    action_type=alt["action_type"],
                    total_value=alt["total_value"],
                    scientific_information_value=alt["normalized_hig"],
                    discovery_value=alt["normalized_disc"],
                    cost_penalty=alt["cost_penalty"],
                    hypothesis_id=dominant_h,
                    contrastive_rationale=contrast,
                )
            )

        # Falsification conditions
        cand_comp = candidate_pool_df[candidate_pool_df["candidate_id"] == top["candidate_id"]][["Au", "Ir", "Rh"]].iloc[0].to_numpy()
        falsification = ensemble.hypotheses[dominant_h].falsification_summary(
            candidate_id=top["candidate_id"],
            action_type=top["action_type"],
            composition=cand_comp,
        )

        action_name = "XRD Characterization" if top["action_type"] == ExperimentActionType.XRD else "SECCM Property Test"
        rationale = (
            f"Selects {action_name} for candidate '{top['candidate_id']}' (Net Scientific Value: {top['total_value']:.3f}). "
            f"Expected Hypothesis Information Gain is {top['raw_hig']:.3f} nats (expected posterior entropy: {top['expected_entropy']:.3f} nats) "
            f"under policy mode '{self.mode.value}'."
        )

        # Supporting evidence string list
        evidence = [
            f"Expected HIG: {top['raw_hig']:.4f} nats",
            f"Property disagreement: {top['property_disagreement']:.6f}",
            f"Structure disagreement: {top['structure_disagreement']:.4f}",
        ]
        for hid, pred in top["predictions"].items():
            if top["action_type"] == ExperimentActionType.PROPERTY:
                evidence.append(f"{hid} predicted k0 mean: {pred.mean[0]:.5f} cm/s (var: {pred.variance[0]:.2e})")
            else:
                evidence.append(f"{hid} predicted XRD emb[0]: {pred.mean[0]:.3f} (var: {pred.variance[0]:.2e})")

        return ActionRecommendation(
            action=act,
            total_value=top["total_value"],
            scientific_information_value=top["normalized_hig"],
            discovery_value=top["normalized_disc"],
            cost_penalty=top["cost_penalty"],
            hypothesis_id=dominant_h,
            rationale=rationale,
            falsification_criterion=falsification,
            supporting_evidence=evidence,
            uncertainty_summary={
                "hypothesis_information_gain": top["raw_hig"],
                "current_entropy": top["current_entropy"],
                "expected_posterior_entropy": top["expected_entropy"],
                "property_disagreement": top["property_disagreement"],
                "structure_disagreement": top["structure_disagreement"],
            },
            alternatives=alternatives,
        )

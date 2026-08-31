from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.datasets.auirh_actions import AuIrRhMultimodalOracle

from src.science.actions import (
    ActionRecommendation,
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
)
from src.science.agents import AgentPerspective, MultiAgentPresentationLayer
from src.science.hypotheses import HypothesisEngine, get_default_hypotheses
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.experiment_policy import NextBestExperimentPolicy
from src.science.scientific_models import PropertySurrogateModel, StructureSurrogateModel
from src.science.xrd_representation import XRDRepresentationExtractor

logger = logging.getLogger(__name__)


class AutonomousDiscoveryEngine:
    """Unified Multimodal Discovery Engine for the AI Scientist Discovery Console.

    Orchestrates the complete closed-loop scientific decision workflow:
    1. Observable candidate landscape & strict offline-oracle firewall.
    2. Dynamic XRD representation & ephemeral surrogate projections.
    3. Structured scientific hypothesis belief updates.
    4. Adaptive Next-Best-Experiment policy scoring with contrastive counterfactuals.
    5. Multi-agent role-based reasoning layer.
    6. Authoritative tamper-evident ledger event logging.
    """

    def __init__(
        self,
        oracle: AuIrRhMultimodalOracle | None = None,
        db_path: Path | str = ":memory:",
        seed: int = 42,
        cost_xrd: float = 1.0,
        cost_property: float = 5.0,
    ) -> None:
        self.seed = seed
        if oracle is None:
            from src.datasets.auirh_actions import AuIrRhMultimodalOracle

            self.oracle = AuIrRhMultimodalOracle()
        else:
            self.oracle = oracle
        self.ledger = ExperimentLedger(db_path=db_path)
        self.xrd_extractor = XRDRepresentationExtractor()
        self.structure_model = StructureSurrogateModel(random_state=seed)
        self.property_model = PropertySurrogateModel(random_state=seed)
        self.hypothesis_engine = HypothesisEngine(get_default_hypotheses())
        self.policy = NextBestExperimentPolicy(
            cost_xrd=cost_xrd,
            cost_property=cost_property,
        )
        self.agent_layer = MultiAgentPresentationLayer()

        self.current_step = 0
        self.total_budget_spent = 0.0
        self.timeline: list[dict[str, Any]] = []
        self._last_recommendation: ActionRecommendation | None = None
        self._last_perspectives: list[AgentPerspective] = []

    def initialize_curated_scenario(
        self,
        n_init_prop: int = 6,
        n_init_xrd: int = 4,
        seed: int | None = None,
    ) -> None:
        """Sets up a deterministic, reproducible initial scientific campaign state.

        Reveals a small initial subset of XRD and property measurements without cherry-picking.
        """
        init_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(init_seed)

        self.reset()
        cand_df = self.oracle.get_candidate_pool()
        all_cids = cand_df["candidate_id"].tolist()

        # Randomly choose initial candidates using fixed seed
        shuffled_cids = list(all_cids)
        rng.shuffle(shuffled_cids)

        init_prop_cids = shuffled_cids[:n_init_prop]
        init_xrd_cids = shuffled_cids[n_init_prop : n_init_prop + n_init_xrd]

        # 1. Execute initial XRD actions
        for cid in init_xrd_cids:
            act = ScientificAction(
                action_id=f"init_xrd_{cid}",
                candidate_id=cid,
                action_type=ExperimentActionType.XRD,
                estimated_cost=self.policy.cost_xrd,
                requested_at_step=0,
            )
            outcome = self.oracle.execute(act)
            self._record_to_ledger(act, outcome)
            self.total_budget_spent += self.policy.cost_xrd
            self.timeline.append(
                {
                    "step": len(self.timeline) + 1,
                    "action_id": act.action_id,
                    "candidate_id": cid,
                    "action_type": "XRD",
                    "cost": self.policy.cost_xrd,
                    "hypothesis_tested": "Initial Baseline",
                    "revealed_summary": f"Peak 2theta: {outcome.revealed_data.get('peak_two_theta', 0.0):.1f}°",
                    "status": "COMPLETED",
                }
            )

        # 2. Execute initial Property actions
        for cid in init_prop_cids:
            act = ScientificAction(
                action_id=f"init_prop_{cid}",
                candidate_id=cid,
                action_type=ExperimentActionType.PROPERTY,
                estimated_cost=self.policy.cost_property,
                requested_at_step=0,
            )
            outcome = self.oracle.execute(act)
            self._record_to_ledger(act, outcome)
            self.total_budget_spent += self.policy.cost_property
            self.timeline.append(
                {
                    "step": len(self.timeline) + 1,
                    "action_id": act.action_id,
                    "candidate_id": cid,
                    "action_type": "PROPERTY",
                    "cost": self.policy.cost_property,
                    "hypothesis_tested": "Initial Baseline",
                    "revealed_summary": f"k0: {outcome.revealed_data.get('k0', 0.0):.5f} cm/s",
                    "status": "COMPLETED",
                }
            )

        self._refit_models_and_update_hypotheses()

    def _refit_models_and_update_hypotheses(self) -> None:
        """Refits surrogate projections and updates hypothesis beliefs from revealed ledger evidence."""
        cand_df = self.oracle.get_candidate_pool()
        obs_df = self.oracle.get_observable_dataset()

        # 1. Fit XRD Representation Extractor on revealed spectra
        revealed_xrd_map = self.oracle._revealed_xrd
        revealed_spectra = [
            out.revealed_data["normalized_intensity"] for out in revealed_xrd_map.values()
        ]
        self.xrd_extractor.fit(revealed_spectra)

        # 2. Extract embeddings for candidates with XRD
        revealed_xrd_cids = list(revealed_xrd_map.keys())
        if revealed_xrd_cids:
            xrd_comps = cand_df[cand_df["candidate_id"].isin(revealed_xrd_cids)][["Au", "Ir", "Rh"]].to_numpy()
            xrd_embs = np.array(
                [self.xrd_extractor.transform(revealed_xrd_map[cid].revealed_data["normalized_intensity"]) for cid in revealed_xrd_cids]
            )
            self.structure_model.fit(xrd_comps, xrd_embs)
        else:
            self.structure_model.fit(np.empty((0, 3)), np.empty((0, 8)))

        # 3. Fit Property Model on revealed properties
        revealed_prop_map = self.oracle._revealed_property
        revealed_prop_cids = list(revealed_prop_map.keys())
        if revealed_prop_cids:
            prop_comps = cand_df[cand_df["candidate_id"].isin(revealed_prop_cids)][["Au", "Ir", "Rh"]].to_numpy()
            prop_targets = np.array([revealed_prop_map[cid].revealed_data["k0"] for cid in revealed_prop_cids])

            # If candidates have both XRD and property, evaluate structural predictive advantage
            joint_cids = [cid for cid in revealed_prop_cids if cid in revealed_xrd_map]
            if len(joint_cids) >= 3:
                joint_comps = cand_df[cand_df["candidate_id"].isin(joint_cids)][["Au", "Ir", "Rh"]].to_numpy()
                joint_targets = np.array([revealed_prop_map[cid].revealed_data["k0"] for cid in joint_cids])
                joint_embs = np.array([self.xrd_extractor.transform(revealed_xrd_map[cid].revealed_data["normalized_intensity"]) for cid in joint_cids])
                self.property_model.fit(prop_comps, prop_targets, embeddings=None)
                eval_dict = self.property_model.evaluate_structure_predictive_advantage(
                    compositions=joint_comps,
                    targets=joint_targets,
                    embeddings=joint_embs,
                )
                struct_adv = eval_dict["structure_advantage_ratio"]
            else:
                self.property_model.fit(prop_comps, prop_targets)
                struct_adv = 0.0
        else:
            self.property_model.fit(np.empty((0, 3)), np.empty((0,)))
            struct_adv = 0.0

        # 4. Update hypothesis evidence
        num_xrd = len(revealed_xrd_map)
        num_prop = len(revealed_prop_map)
        self.hypothesis_engine.update_evidence(
            num_xrd=num_xrd,
            num_prop=num_prop,
            structure_advantage_ratio=struct_adv,
            structure_novelty_mean=0.35 if num_xrd < 8 else 0.15,
            structure_residual_norm=0.25,
            property_residual_norm=0.20,
        )

    def _record_to_ledger(self, action: ScientificAction, outcome: ExperimentOutcome) -> None:
        """Appends experiment record to authoritative ledger via valid lifecycle transitions."""
        cand_df = self.oracle.get_candidate_pool()
        row = cand_df[cand_df["candidate_id"] == action.candidate_id].iloc[0]

        proposal_rec = ScientificExperimentRecord(
            experiment_id=action.action_id,
            candidate_id=action.candidate_id,
            dataset_name="Au-Ir-Rh_Multimodal_Demo",
            stage=ExperimentStage.PROPOSED,
            pre_experiment_features={"Au": float(row["Au"]), "Ir": float(row["Ir"]), "Rh": float(row["Rh"])},
            candidate_variables={"Library": str(row["Library"]), "Area": int(row["Area"])},
            proposal_metadata=action.to_dict(),
            provenance=outcome.provenance,
        )
        self.ledger.record_proposal(proposal_rec)

        self.ledger.append_transition(
            experiment_id=action.action_id,
            new_stage=ExperimentStage.EXECUTED,
            event_type="EXPERIMENT_EXECUTED",
            delta_payload={},
        )

        delta = (
            {"characterization": outcome.revealed_data}
            if action.action_type == ExperimentActionType.XRD
            else {"performance": outcome.revealed_data}
        )
        self.ledger.append_transition(
            experiment_id=action.action_id,
            new_stage=ExperimentStage.COMPLETED,
            event_type="EXPERIMENT_COMPLETED",
            delta_payload=delta,
        )

    def propose_next_experiment(self) -> tuple[ActionRecommendation, list[AgentPerspective]]:
        """Evaluates policy and multi-agent layer to recommend the next scientific experiment."""
        cand_df = self.oracle.get_candidate_pool()
        observed_xrd = set(self.oracle._revealed_xrd.keys())
        observed_prop = set(self.oracle._revealed_property.keys())

        recommendation = self.policy.recommend_next_experiment(
            candidate_pool_df=cand_df,
            observed_xrd_ids=observed_xrd,
            observed_property_ids=observed_prop,
            structure_model=self.structure_model,
            property_model=self.property_model,
            hypothesis_engine=self.hypothesis_engine,
            step=self.current_step,
        )

        perspectives = self.agent_layer.generate_perspectives(
            recommendation=recommendation,
            hypothesis_engine=self.hypothesis_engine,
            num_xrd_revealed=len(observed_xrd),
            num_property_revealed=len(observed_prop),
            total_candidates=len(cand_df),
        )

        self._last_recommendation = recommendation
        self._last_perspectives = perspectives
        return recommendation, perspectives

    def execute_experiment(self, action: ScientificAction | None = None) -> dict[str, Any]:
        """Executes the recommended (or specified) scientific action via the oracle."""
        act = action or (self._last_recommendation.action if self._last_recommendation else None)
        if act is None:
            raise ValueError("No action provided or recommended to execute.")

        # Capture before-state beliefs
        before_beliefs = {hid: h.belief_score for hid, h in self.hypothesis_engine.hypotheses.items()}

        outcome = self.oracle.execute(act)
        self._record_to_ledger(act, outcome)

        self.current_step += 1
        self.total_budget_spent += act.estimated_cost

        # Re-fit models and update beliefs
        self._refit_models_and_update_hypotheses()

        # Capture after-state beliefs
        after_beliefs = {hid: h.belief_score for hid, h in self.hypothesis_engine.hypotheses.items()}
        belief_deltas = {hid: after_beliefs[hid] - before_beliefs[hid] for hid in after_beliefs}

        summary = {
            "step": self.current_step,
            "action": act.to_dict(),
            "outcome": outcome.to_dict(),
            "budget_spent": self.total_budget_spent,
            "before_beliefs": before_beliefs,
            "after_beliefs": after_beliefs,
            "belief_deltas": belief_deltas,
            "best_observed_k0": self.oracle.get_revealed_state_summary()["best_observed_k0"],
        }

        # Append to timeline
        revealed_str = (
            f"XRD Peak 2theta: {outcome.revealed_data.get('peak_two_theta', 0.0):.1f}°"
            if act.action_type == ExperimentActionType.XRD
            else f"Measured k0: {outcome.revealed_data.get('k0', 0.0):.5f} cm/s"
        )
        self.timeline.append(
            {
                "step": len(self.timeline) + 1,
                "action_id": act.action_id,
                "candidate_id": act.candidate_id,
                "action_type": act.action_type.value,
                "cost": act.estimated_cost,
                "hypothesis_tested": act.metadata.get("hypothesis_id", "H1"),
                "revealed_summary": revealed_str,
                "status": "COMPLETED",
            }
        )

        return summary

    def get_landscape_dataframe(self) -> pd.DataFrame:
        """Returns observable landscape DataFrame for UI visualization."""
        cand_df = self.oracle.get_candidate_pool().copy()
        all_comps = cand_df[["Au", "Ir", "Rh"]].to_numpy(dtype=np.float64)

        # Predictions
        _, struct_stds = self.structure_model.predict(all_comps)
        prop_means, prop_stds = self.property_model.predict(all_comps)

        cand_df["struct_uncertainty"] = struct_stds
        cand_df["predicted_k0"] = prop_means
        cand_df["prop_uncertainty"] = prop_stds

        # Status categorization
        statuses = []
        measured_k0s = []
        for _, row in cand_df.iterrows():
            cid = row["candidate_id"]
            has_xrd = self.oracle.is_xrd_observed(cid)
            has_prop = self.oracle.is_property_observed(cid)

            if has_xrd and has_prop:
                statuses.append("Both XRD & Property")
            elif has_xrd:
                statuses.append("XRD Characterized")
            elif has_prop:
                statuses.append("Property Tested")
            else:
                statuses.append("Unobserved")

            if has_prop:
                measured_k0s.append(self.oracle._revealed_property[cid].revealed_data["k0"])
            else:
                measured_k0s.append(np.nan)

        cand_df["status"] = statuses
        cand_df["measured_k0"] = measured_k0s
        return cand_df

    def reset(self) -> None:
        """Resets the engine and oracle for fresh replay."""
        self.oracle.reset()
        self.ledger = ExperimentLedger(db_path=":memory:")
        self.xrd_extractor = XRDRepresentationExtractor()
        self.structure_model = StructureSurrogateModel(random_state=self.seed)
        self.property_model = PropertySurrogateModel(random_state=self.seed)
        self.hypothesis_engine = HypothesisEngine(get_default_hypotheses())
        self.current_step = 0
        self.total_budget_spent = 0.0
        self.timeline.clear()
        self._last_recommendation = None
        self._last_perspectives.clear()
